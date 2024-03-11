import copy
import json
import random
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Type

import yaml
from neuri.constrinf import _process_record
from neuri.constrinf.constr import Constraint, convert_constr_to_executable, convert_dtypes_to_z3s
from neuri.constrinf.errmsg import ErrorMessage, is_similar, map_error_messages_to_clusters_dynamic
from neuri.constrinf.executor import Executor
from neuri.constrinf.inferencer import Inferencer
import hydra
from omegaconf import DictConfig

from neuri.backends.factory import BackendFactory
from neuri.constrinf.parser import segment_constr, parse_from_raw_txt
from neuri.constrinf.prompter import Prompter
from neuri.constrinf.synthesizer import Synthesizer
from neuri.constrinf.util import formatted_dict
from neuri.error import WrongInferenceError
from neuri.gir import GraphIR
from neuri.logger import TRAIN_LOG, AUTOINF_LOG, TRAIN_LOG
from neuri.macro import NNSMITH_BUG_PATTERN_TOKEN
from neuri.materialize import Model
from neuri.util import mkdir, parse_timestr, set_seed

# Status.csv
# Each line represents the status of one generation.
# META: seed,stat,tgen,tsmt,tsave,trun,elapsed_s
# stat can be one of:
#  - "ok": no error.
#  - "fail": invalid testcase.
#  - "bug": bug found.
SPLITTER = "\n"
class Scores(Dict): pass
class Record(Dict): pass
def transform_record_for_saving(record: Record) -> dict:
    """
    Transform the record dictionary to the original format expected for saving.

    Args:
        record (dict): The modified record dictionary.

    Returns:
        dict: The transformed record dictionary suitable for saving.
    """
    transformed = {}
    for key, value in record.items():
        if key == 'args':
            transformed[key] = {k : v for k, v in record[key].items() if k not in ["value", "dtype_obj"]}
        elif key in ["outputs"] :
            pass
        elif key == "rules" :
            transformed[key] = list(value)
        else:
            transformed[key] = value
    return transformed

def save_record(record: dict, path: str) -> None:
    """
    Save a given record dictionary to a YAML file, ensuring that the dictionary is
    transformed back to its original expected format before saving.

    Args:
        record (dict): The record dictionary to be saved.
        path (str): The file path where the record should be saved.

    Raises:
        FileNotFoundError: If the directory specified in the path does not exist.
        Exception: For any unexpected errors during the save operation.
    """
    # Transform the record back to the expected format
    record = copy.deepcopy(record)
    record = transform_record_for_saving(record)

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        # raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    with open(path, 'w') as file:
        yaml.dump(record, file)

    TRAIN_LOG.debug(f"Record saved successfully to {path}.")

class StatusCollect:
    def __init__(self, root, resume=False, op_usage=False):
        self.root = Path(root)
        stat_fname = self.root / "status.csv"
        self.n_infer = 0
        self.n_synthesized = 0 
        self.cur_high_score = 0
        self.score_history = []
        self.n_bugs = 0
        self.n_fail_make_test = 0
        self.n_testcases = 0
        self._elapsed_s = 0
        self.infer_history = {}
        if (
            os.path.exists(self.root) and resume
        ):  # parse n_bugs, n_fail_make_test, n_testcases.
            with open(stat_fname, "r") as f:
                lines = f.readlines()
            for line in lines[1:]:
                if line == "":
                    continue
                tokens = line.split(",")
                stat = tokens[1]
                self._elapsed_s = float(tokens[-1])
                if stat == "bug":
                    self.n_bugs += 1
                elif stat == "fail":
                    self.n_fail_make_test += 1
                elif stat != "ok":
                    raise ValueError(f"Unknown stat: {stat}")
                self.n_testcases += 1
            TRAIN_LOG.info(f"Resuming from {self.n_testcases} testcases.")
        else:
            mkdir(self.root)
            with open(stat_fname, "w") as f:
                f.write("seed,stat,tgen(ms),tsmt(ms),tsave(ms),trun,elapsed(s)\n")

        self.stat_file = open(
            self.root / "status.csv", "a", buffering=1024
        )  # 1KB buffer
        self.tik = None

        self.op_usage = op_usage
        if self.op_usage:
            self.op_used_file = open(self.root / "op_used.txt", "a", buffering=128)
            self.op_rej_file = open(self.root / "op_rej.txt", "a", buffering=128)

    @property
    def elapsed_s(self) -> float:
        if self.tik is None:
            self.tik = time.time()
        else:
            tok = time.time()
            self._elapsed_s += tok - self.tik
            self.tik = tok
        return self._elapsed_s

    def record(self, seed, stat, tgen, tsmt=0, trun=0, tsave=0):
        self.n_testcases += 1
        if stat == "bug":
            self.n_bugs += 1
        elif stat == "fail":
            self.n_fail_make_test += 1
        elif stat != "ok":
            raise ValueError(f"Unknown stat: {stat}")
        self.stat_file.write(
            f"{seed},{stat},{tgen},{tsmt},{tsave},{trun},{self.elapsed_s:.3f}\n"
        )
        self.stat_file.flush()
        TRAIN_LOG.info(
            f"tgen={tgen:.1f}ms, tsmt={tsmt:.1f}ms, trun={trun:.1f}ms, tsave={tsave:.1f}ms"
        )

class TrainingLoop:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        
        self.cfg = cfg
        self.inferencer = Inferencer(cfg['llm']['settings'])
        cmpwith = cfg["cmp"]["with"]
        if cfg["backend"]["type"] == "tflite" and (
            cmpwith is None or cmpwith["target"] != "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if (
            cfg["train"]["crash_safe"]
            and cfg["backend"]["type"] == "xla"
            and cfg["backend"]["target"] == "cuda"
        ) or (
            cmpwith is not None
            and cmpwith["type"] == "xla"
            and cmpwith["target"] == "cuda"
        ):
            raise ValueError(
                "Please set `fuzz.crash_safe=false` for XLA on CUDA. "
                "Also see https://github.com/ise-uiuc/nnsmith/blob/main/doc/known-issues.md"
            )

        self.n_infer = int(cfg["train"]["n_infer_per_round"])

        self.factory = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
        )

        model_cfg = self.cfg["model"]
        self.constr_target_map = {}
        self.ModelType = Model.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )

        self.executor : Executor = Executor(self.ModelType, parallel = cfg["train"]["parallel"])
        self.train_list = self.get_train_list()
        TRAIN_LOG.info(
            f"{len(self.train_list)} opsets wait for inferring"
        )
        self.stat = {}
        # self.save_test = cfg["train"]["save_test"]
        # resume = cfg["train"]["resume"]
        # self.status = StatusCollect(
        #     cfg["train"]["root"],
        #     resume=resume,
        #     op_usage=(
        #         cfg["mgen"]["max_nodes"] == 1 and cfg["mgen"]["method"] == "neuri-i"
        #     ),
        # )
        # if isinstance(self.save_test, str) and not (
        #     os.path.exists(self.save_test) and resume
        # ):  # path of root dir.
        #     TRAIN_LOG.info(f"Saving all intermediate testcases to {self.save_test}")
        #     mkdir(self.save_test)

    def init_stat(self) :
        self.stat = {
            "exec" : 0,
            "llm" : 0,
            "smt" : 0,
            "save" : 0
        }

    def get_train_list(self, path = "/artifact/data/torch_overall_apis.json") -> List[str]:
        """
        get operator yaml file path
        """
        def sort_key(file_name):
            base_name, index_with_extension = file_name.split('-')[0], file_name.split('-')[1]
            index = index_with_extension.replace('.yaml', '')
            return base_name, index
        with open(path, "r") as f:
            data = json.load(f)

        li = []
        root_path = self.cfg["train"]["record_path"] 
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".yaml"):
                    name = file.split(".")[0].split("-")[0]
                    if name in data :
                        li.append(os.path.join(root, file))
        return sorted(li, key=sort_key)
        # li = []
        # root_path = self.cfg["train"]["record_path"] 
        # for root, _, files in os.walk(root_path):
        #     for file in files:
        #         if file.endswith(".yaml"):
        #             li.append(os.path.join(root, file))
        # return li
    
    def get_pass_rate_and_err_msgs(self, record, ntimes) -> Tuple[float, ErrorMessage]:
        success_count = 0
        error_messages = {}
        raw_err_msgs = []
        copied_record = copy.deepcopy(record)
        executable_constr = convert_constr_to_executable(copied_record) # unactivated
        results = self.executor.execute(record = copied_record, 
                                        constraints = executable_constr,
                                        ntimes = ntimes, 
                                        noise = self.cfg["train"]["noise"],
                                        allow_zero_length_rate = self.cfg["train"]["allow_zero_length_rate"],
                                        allow_zero_rate = self.cfg["train"]["allow_zero_rate"],
                                        num_of_try = self.cfg["train"]["num_of_try"]
                                        )
        
        for result in results:
            if result is None :
                continue
            success, error_instance = result
            if success:
                success_count += 1
            else:
                msg_key = error_instance.get_core_msg()
                error_messages[msg_key] = error_instance
                raw_err_msgs.append(msg_key)
        if raw_err_msgs :
            dynamic_cluster_mapping = map_error_messages_to_clusters_dynamic(raw_err_msgs)
            sorted_cluster_mapping = dict(sorted(dynamic_cluster_mapping.items(), key=lambda item: len(item[1]), reverse=True))
            success_rate = success_count / (success_count + len(error_messages))
            sorted_error_instances = [error_messages[list(li)[0]] for li in sorted_cluster_mapping.values()]
        else :
            success_rate = 1
            sorted_error_instances = []

        if sorted_error_instances:
            most_frequent_err = sorted_error_instances[0]
            distributions = {messages[0] : len(messages) for _, messages in sorted_cluster_mapping.items()}
            TRAIN_LOG.debug(f"Current error distribution :\n{formatted_dict(distributions)}")
            if self.is_special_err_msg(most_frequent_err):
                raise Exception("Special error message encountered. Cannot proceed.")
        else:
            TRAIN_LOG.info("No error messages encountered.")

        # Return success rate and the sorted list of error message instances
        return success_rate*100, sorted_error_instances
    
    def select_train_op(self) :
        if self.train_list :
            return self.train_list.pop(0)
        else :
            return None 

    def parse_and_generate_rules(self, raw_infered : str, target : ErrorMessage, arg_names : List[str]) -> List[Constraint]:
        """
        Parse LLM response and generate rules.
        Filter ill-formed rules and return the valid ones.
        """
        generated = []
        rules = []
        infered, cot = parse_from_raw_txt(raw_infered)
        dtypes = target.get_dtypes(arg_names)
        for rule_txt in infered.split(';'):
            generated.append(rule_txt.strip())
            generated.extend(segment_constr(rule_txt))
        
        for rule_txt in generated:
            if rule_txt :
                rule = Constraint(rule_txt, cot, target, arg_names, dtypes)
                if not rule.is_error() and rule.check() :
                    rules.append(rule)
        return rules

    def update_tolerance_and_history(self, result : Tuple[Scores, Constraint], tolerance, highest_prev, prev_answer):
        """
        Update the training session's state based on the latest rule evaluation results.
        
        :param results: The results from the latest round of rule evaluations.
        :param tolerance: The current tolerance level.
        :param highest_prev: The highest score achieved so far.
        :param prev_answer: The current best rule based on previous evaluations.
        :return: A tuple containing updated tolerance, highest_prev, and prev_answer.
        """
        if highest_prev["overall_score"] >= result[0]["overall_score"] :
            # No new rules to evaluate or no improvement found, increase tolerance
            tolerance += 1
        else:
            # Extract the best result from the current evaluation
            best_current_score, best_constr = result
            # Found a better rule, reset tolerance and update history
            TRAIN_LOG.info(f'New highest score: from {highest_prev["overall_score"]} to {best_current_score["overall_score"]} with rule {best_constr.txt}')
            highest_prev = best_current_score
            prev_answer = best_constr
            tolerance = 0  # Reset tolerance since we found an improvement
        
        # Check if an optimal rule has been found
        solved = highest_prev["overall_score"] == 100
        return solved, tolerance, highest_prev, prev_answer

    def finalize_training_session(self, 
                                  highest_prev : Dict[str, float], 
                                  record : Dict[str, Any], 
                                  constr_list : List[Tuple[Constraint, Scores]],
                                  synthesizer : Synthesizer, 
                                  prev_answer : Constraint = None) -> bool:
        """
        Finalize the training session, handling the selection of the most optimal rule.
        
        :param highest_prev: The highest score achieved during training.
        :param synthesizer: The synthesizer instance used during training.
        :param prev_answer: The rule corresponding to the highest score.
        :return: Boolean indicating whether a satisfactory rule was found.
        """
        # Log the tried rules and their outcomes
        TRAIN_LOG.info(f"Tried rules:\n{SPLITTER.join(map(str, synthesizer.tried))}")
        
        if highest_prev["overall_score"] == 0:
            # No improvement found
            TRAIN_LOG.info("No improvement found.")
            return False
        elif highest_prev["overall_score"] == 100:
            # A rule with perfect score found
            TRAIN_LOG.info(f"Perfect rule found: {prev_answer.txt}")
            self.handle_solved_rule(prev_answer, highest_prev, record, constr_list)
            return True
        else:
            # Select the best rule based on the training outcome
            if synthesizer.non_FP:
                # Select the first non-false positive rule if available
                best_rule = synthesizer.non_FP[0][1]
            else:
                # Fallback to the best seed rule if the precision threshold is met
                if synthesizer.seeds[0][0]["precision"] < self.cfg["train"]['precision_threshold']:
                    TRAIN_LOG.info("Best rule does not meet the precision threshold.")
                    return False
                best_rule = synthesizer.seeds[0][1]
            
            TRAIN_LOG.info(f"Applying best found rule: {best_rule.txt}")
            self.handle_solved_rule(best_rule, highest_prev, record, constr_list) ## save rule to record
            return True   
        
    def get_only_acc_save_path(self, record_path) :
        ## change constraints -> only_acc
        new_path = record_path.replace("records", "only_acc")
        return new_path
    
    def get_retrain_list(self, record, constr_list : List[Tuple[Constraint, Scores]]) -> List[Tuple[Constraint, Scores]] :
        # pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record)
        data = constr_list[:]
        for constr, scores in self.get_constrs_from_rules(record) :
            if constr.txt in [c.txt for c, _ in data] :
                continue 
            data.append((constr, scores))
        TRAIN_LOG.info(f"start retraining with {len(data)} constraints")
        return data
    
    @staticmethod
    def get_constrs_from_rules(record : Record) -> List[Any] :
        if record.get("rules", None) is None :
            record["rules"] = []
        return [(Constraint.load(rule_data), scores) for rule_data, scores in record["rules"]]
    
    def pop_constr_from_record(self, txt : str, record) :
        for i, (rule_data, _) in enumerate(record["rules"]) :
            if rule_data["txt"] == txt :
                popped = record["rules"].pop(i)
                return (Constraint.load(popped[0]), popped[1])
        raise ValueError(f"no such constraint in record : {txt}")

    def is_trainable(self, record : Dict[str, Any]) :
        return (
            record is not None and \
            record.get("error", None) is None
        )
    def runs(self) :
        n_debug = 50
        n_func = 0
        # while True :
        while n_func < n_debug :
            n_func+=1
            record_path = self.select_train_op()
            if record_path is None :
                break
            op_record = _process_record(record_path)
            TRAIN_LOG.info(f"Start infering {op_record['name']}")
            if self.is_trainable(op_record) :
                try :
                    self.run(op_record, record_path)
                    self.finalize_infering(op_record)
                except :
                    TRAIN_LOG.error(f"{traceback.format_exc()}")
            else :
                TRAIN_LOG.warning(f"Record is not trainable : {op_record}")
    def finalize_infering(self, record) :
        pass_rate, unsolved = self.get_pass_rate_and_err_msgs(record, self.cfg["train"]["eval_asset"])
        self.update_pass_rate(record, pass_rate)
        TRAIN_LOG.info(
f"""{record['name']} infered finished\n
constrs : {record['rules']}\n
final_pass_rate : {record['pass_rate']}\n
unsolved_err_msgs : {[e.dump() for e in unsolved]}
            """
        )  
        self.inferencer.finalize()

    def load_constr_target_map_from_record(self, record) :
        """load constr target map from record"""
        constr_target_map = {}
        # for constr in self.get_constrs_from_rules(record) :
        #     constr_target_map.update({constr["target"] : (constr["scores"], Constraint(constr["txt"], constr["cot"], constr["target"], record["args"]["name"], record["args"]["dtype"]))})
        return constr_target_map
    
    def update_pass_rate(self, record, pass_rate = None) :
        if pass_rate is None :
            pass_rate, unsolved = self.get_pass_rate_and_err_msgs(record, self.cfg["train"]["eval_asset"])
        TRAIN_LOG.info(f"pass_rate : {pass_rate}")
        record["pass_rate"] = pass_rate

    def save_only_acc(self, record, record_path) :
        save_path = self.get_only_acc_save_path(record_path)
        if os.path.exists(save_path) :
            return 
        save_record(record, save_path)
        
    def run(self, op_record, record_path):

        n_try = 0
        pass_rate = 0
        constr_list : Tuple[List, Dict[str, Any]][Constraint] = []
        TRAIN_LOG.info(f"Start training {op_record['name']}")
        while n_try < self.cfg["train"]["n_try"] :
            pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record, self.cfg["train"]["eval_asset"])
            self.update_pass_rate(op_record, pass_rate)
            if pass_rate >= self.cfg["train"]["precision_threshold"] :
                TRAIN_LOG.info(f"pass_rate is over {self.cfg['train']['precision_threshold']}")
                break
            if sorted_err_instances :
                succeed = self.train(op_record, sorted_err_instances[0], constr_list, mode="acc")
                if succeed :
                    save_record(op_record, record_path)
                else :
                    TRAIN_LOG.warning(f"Failed to train {sorted_err_instances[0].get_core_msg()}")
                    break
            else :
                TRAIN_LOG.info(f"No error messages encountered.")
                break
        
        self.save_only_acc(op_record, record_path)
        queue = self.get_retrain_list(op_record, constr_list)
        while queue :
            constr, scores = queue.pop()
            self.pop_constr_from_record(constr.txt, op_record)
            succeed = self.train(op_record, constr.target, [], mode="f1", seeds = [(scores, constr)])
            if succeed :
                save_record(op_record, record_path)
            else :
                break
    
    def fix_record_attrs(self, record, target : ErrorMessage) :
        """
        fix dtypes and rules
        multiple dtypes -> single dtype
        txt rule -> executable rule(unactivated)
        """
        copied = copy.deepcopy(record)
        copied["rules"] = convert_constr_to_executable(copied)
        copied["args"]["dtype_obj"] = [[d] for d in target.get_dtypes(copied["args"]["name"])]
        return copied
    
    def train(self, 
              orig_record, 
              target : ErrorMessage,
              constr_list : List[Tuple[Constraint, Dict[str, Any]]],  
              mode : Literal["acc", "f1"], 
              seeds = []):
        # Initialize training session
        self.inferencer.change_to_gpt3()
        solved = False
        infer_times = 0
        prev_answer = None
        highest_prev : Dict[str, float] = {"overall_score" : 0, "precision" : 0, "recall" : 0, "f1_score" : 0}
        tolerance = 0
        new_rules : List[Constraint] = []
        record = self.fix_record_attrs(orig_record, target) # deep copy and fix dtypes and rules
        prompter = Prompter(record)
        synthesizer = Synthesizer(target, self.executor, record, self.cfg["train"])
        # Synthesizer responsible for filtering, evaluating, and finding the best rule
        synthesizer.set_mode(mode)
        if seeds : 
            TRAIN_LOG.info(f"Start training with : {seeds[0][1]}")
            if seeds[0][0]["f1_score"] == 100 :
                self.update_record_constr(seeds[0][1], orig_record, seeds[0][0])
                return True
            synthesizer.save_state(seeds)
        while self.cfg["train"]['infer_asset_per_epoch'] >= infer_times and not solved:
            if tolerance >= self.cfg["train"]['tolerance']//2:
                self.inferencer.change_to_gpt4()
                break
            
            context, prompts = prompter.gen([target.get_core_msg()], 
                                    args_values=target.get_values_map(),
                                    func_name=record["name"],
                                    num_of_ex=random.randint(2, 3),
                                    prev_answer=prev_answer)
            
            raw_infered = self.inferencer.inference(prompts, context) 
#             raw_infered = """Error is triggered because the input tensor A must have at least 2 dimensions, but the provided value is a 1-dimensional tensor. To prevent this error from occurring again, we can generate constraints that ensure the input tensor has at least 2 dimensions. We can define the constraint as follows:
# ```len(input.shape) >= 2```""" # for debugging
            infer_times += 1
            
            new_rules = self.parse_and_generate_rules(raw_infered, 
                                                      target, 
                                                      record["args"]["name"]
                                                      )
            result = synthesizer.run(new_rules)
            if result is not None :
                solved, tolerance, highest_prev, prev_answer = self.update_tolerance_and_history(result, tolerance, highest_prev, prev_answer)
        return self.finalize_training_session(highest_prev, orig_record, constr_list, synthesizer, prev_answer)
    
    def update_record_constr(self, constr : Constraint, record, scores) :
        record["rules"].append(
            [constr.dump(), scores]
        )
    def handle_solved_rule(self, 
                           constr : Constraint, 
                           scores : Dict[str, Any], 
                           record : Dict[str, Any],
                           constr_list : List[Tuple[Constraint, Dict[str, Any]]],
                           ) : 
        constr_list.append((constr, scores))
        self.update_record_constr(constr, record, scores)

    def is_special_err_msg(self, errmsg : ErrorMessage) -> bool:
        return errmsg.error_type in [TypeError]


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    try :
        TrainingLoop(cfg).runs()
    except :
        TRAIN_LOG.error(f"{traceback.format_exc()}")


if __name__ == "__main__":
    main()