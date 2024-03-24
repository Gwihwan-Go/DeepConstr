import concurrent.futures
import functools
from multiprocessing import Manager, Pool
import multiprocessing
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple
from neuri.abstract.dtype import AbsTensor
from neuri.autoinf.instrument.op import OpInstance
from neuri.constrinf import record_args_info
from neuri.error import InternalError
from neuri.logger import MGEN_LOG
from neuri.specloader.smt import DEFAULT_DTYPE_CONSTR, gen_val
from neuri.constrinf.errmsg import ErrorMessage

NOERR_MSG = "no error"

_gloabl_constraints = []
_dtype_constrs_executable = []

def set_global_constraints(constraints) :
    global _gloabl_constraints
    _gloabl_constraints = constraints

def clear_global_constraints() :
    global _gloabl_constraints
    _gloabl_constraints = []

def worker(model, record, noise=0.8, allow_zero_length_rate=0.1, allow_zero_rate=0.1, num_of_try=30):
    chosen_dtype = {}
    concretized_values = {}
    dtype_constrs = []
    for i_arg, arg_name in enumerate(record['args']['name']):
        if record['args']['dtype_obj'][i_arg] is None :
            chosen_dtype[arg_name] = None
            MGEN_LOG.debug(f"Unidentiable dtype for {arg_name} : {record['args']['dtype'][i_arg]}")
        elif len(record['args']['dtype_obj'][i_arg]) > 0:
            chosen_dtype[arg_name] = random.choice(record['args']['dtype_obj'][i_arg])
        else:
            chosen_dtype[arg_name] = record['args']['dtype_obj'][i_arg]
        if isinstance(chosen_dtype[arg_name], AbsTensor) :
            dtype_constrs.append(_dtype_constrs_executable(arg_name))

    values = gen_val(
                num_of_try,
                chosen_dtype, 
                _gloabl_constraints, # constraints
                noise_prob=noise,
                allow_zero_length_rate=allow_zero_length_rate,
                allow_zero_rate=allow_zero_rate,
                api_name=record['name'],
                dtype_constrs=dtype_constrs,
            )
    if values is None : 
        return None
    record_args_info(record, values)
    inst = OpInstance(record)
    for key in values:
        if isinstance(values[key], list):
            concretized_values[key] = [v.concretize(inst.input_symb_2_value, only_shape=True) if hasattr(v, 'concretize') else v for v in values[key]]
        else :
            concretized_values[key] = values[key].concretize(inst.input_symb_2_value, only_shape=True) if hasattr(values[key], 'concretize') else values[key]
    MGEN_LOG.debug(f"Concretized values of {record['name']}: {concretized_values}")
    try:
        # Assuming record_args_info is a function to log or record argument info
        # self.record_args_info(record, values)  # Placeholder for actual logging or recording
        res_or_bug, abs_ret_list = model.execute_op(inst)
        # if res_or_bug == NotImplemented :
        #     err_instance = ErrorMessage("NotImplemented", "", concretized_values, chosen_dtype, package=model.package)
        #     err_instance.error_type = NotImplementedError
        #     return False, err_instance
        return True, ErrorMessage(NOERR_MSG, "", concretized_values, chosen_dtype, package=model.package)  # Assuming execution success
    except Exception as e:
        error_instance = ErrorMessage(e, traceback.format_exc(), concretized_values, chosen_dtype, package=model.package)
        assert isinstance(error_instance, ErrorMessage)
        return False, error_instance  # Return error state and message

def worker_wrapper(worker_fn, return_dict, index, *args, **kwargs):
    try:
        result = worker_fn(*args, **kwargs)
    except Exception as e:
        err_instance = ErrorMessage(InternalError(), traceback.format_exc(), {}, {})
        MGEN_LOG.error(f"Err execute: {e}, maybe child process core dumped")
        result = [False, err_instance]
    return_dict[index] = result

class Executor:
    def __init__(self, model, parallel=8) :
        global _dtype_constrs_executable
        self.model = model
        self.parallel = parallel
        _dtype_constrs_executable = DEFAULT_DTYPE_CONSTR.get(self.model.package)

    def execute(self, ntimes, constraints, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        MGEN_LOG.info(f"Executing {ntimes} times")
        set_global_constraints(constraints) # to be used in worker(parallel execution)
        res = self.parallel_execute(ntimes, *args, **kwargs) \
            if self.parallel != 1 else self._execute(ntimes, *args, **kwargs)
        clear_global_constraints()
        return res
    def _execute(self, ntimes, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        results = []
        worker_fn = functools.partial(worker, self.model, *args, **kwargs)
        for _ in range(ntimes):
            res = worker_fn()
            if res is None :
                results.append(res)
            else :
                results.append(res)
        return results
    def parallel_execute(self, ntimes, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        # def execute_in_parallel(worker, self, ntimes, num_of_task_per_process, *args, **kwargs):
        num_of_task_per_process = ntimes // self.parallel
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        worker_fn = functools.partial(worker, self.model, *args, **kwargs)
        processes = []
        
        for i in range(ntimes):
            p = multiprocessing.Process(target=worker_wrapper, args=(worker_fn, return_dict, i) + args, kwargs=kwargs)
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join(num_of_task_per_process * 2)  # Timeout parameter
        
        # Handle processes that did not finish in time
        for p in processes:
            if p.is_alive():
                p.terminate()  # Terminate process
                err_instance = ErrorMessage(TimeoutError(), "Process exceeded timeout.", {}, {})
                MGEN_LOG.error(f"TIMEOUT error in execute: Process exceeded timeout.")
                return_dict[processes.index(p)] = [False, err_instance]
        
        results = [return_dict[i] for i in range(len(return_dict))]
        return results
    def _parallel_execute(self, ntimes, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        """
        ctypes pickling problem.
        To support parallel execution, 
        Constr is converted to executable(Callable)
        dtypes are converted to z3 types
        Executes the given operation (opinstance) multiple times in parallel using multicore processing.
        Classifies the program result (success or error) and deals with error messages along with argument values.
        
        Args:
        - opinstance: The operation instance to be executed.
        - ntimes: The number of times to execute the operation.
        
        Returns:
        A tuple (success_rate: float, error_messages: dict) where success_rate is the ratio of successful executions
        and error_messages is a dictionary mapping error messages to their corresponding argument values.
        """
        results = []
        num_of_task_per_process = ntimes // self.parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel) as executor:
            # Generate a list of future tasks
            worker_fn = functools.partial(worker, self.model, *args, **kwargs)
            futures = [executor.submit(worker_fn) for _ in range(ntimes)]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=num_of_task_per_process*2)
                results.append(result)
            except TimeoutError:
                err_instance = ErrorMessage(TimeoutError(), traceback.format_exc(), {}, {})
                MGEN_LOG.error(f"TIMEOUT error in execute: {e}")
                results.append([False, err_instance])
            except Exception as e:
                err_instance = ErrorMessage(InternalError(), traceback.format_exc(), {}, {})
                MGEN_LOG.error(f"Error in execute: {e}, maybe child process core dumped")
                results.append([False, err_instance])
        return results