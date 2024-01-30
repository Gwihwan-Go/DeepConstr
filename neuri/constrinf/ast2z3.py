import ast
from functools import partial
import itertools
import random
import string
import traceback
import z3
from typing import *
from neuri.abstract.dtype import AbsDType, AbsIter, DType
from neuri.constrinf.ast_tool import *
from neuri.constrinf.z3const import *
from neuri.error import IncompatiableConstrError

def get_bool_operator(astop : str): 
    if is_same_ast_name(astop, ast.And):
        return z3.And
    elif is_same_ast_name(astop, ast.Or):
        return z3.Or
    elif is_same_ast_name(astop, ast.Not):
        return z3.Not
    else:
        raise ValueError(f"Unknown operator: {astop}")
    
def random_gen_name() : 
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))

def is_tensor(arg) : 
    return hasattr(arg, 'dtype') or (hasattr(arg, "decl") and hasattr(arg.decl(), 'dtype'))

def is_dtype_constant(arg) :
    from neuri.specloader.materalize import STR_TO_ABS
    return isinstance(arg, str) and arg.lower() in STR_TO_ABS.keys()

def get_dtype_z3_obj(arg, is_tensor_dtype=False) : 
    from neuri.specloader.materalize import STR_TO_ABS
    dtypeobj = STR_TO_ABS[arg.lower()] 
    # if is_tensor_dtype and hasattr(dtypeobj, "z3_const") :
    #     # Tensor dtype arg_name would not be changed dtype to int, or list[int], or bool.
    #     # dtype would be retrieved with args_type_dict that has been given when generating the rule.
    #     # Therefore, the type_actvated new dtype would not be shown here.
    #     if dtypeobj.z3_const() is not None :
    #         return [dtype.z3_const() for dtype in dtypeobj.z3_const()]
    return dtypeobj.z3_const()

def merge_constr(constrs) : 
    return z3.And(constrs)

def dict_combinations(input_dict):
    # Create a list of tuples where each tuple is (key, option)
    keys = input_dict.keys()
    
    # Generate all combinations
    try :
        all_combinations = list(itertools.product(*[set(vals) for vals in input_dict.values()]))
    except : 
        print(input_dict)
    # Create a list of dictionaries for each combination
    result = []
    for combination in all_combinations:
        result.append(dict(zip(keys, combination)))

    return result

FLAGS = ["must_iter", "must_int", "must_not_iter", "must_str"]

class Ast2z3(z3funcs) : 
    
    def __init__(self, arg_names, dtypes, txt, func_name) -> None : 
        super().__init__()
        self.need_hijack = False
        self.func_name = func_name
        self.txt = txt
        self.arg_names = arg_names
        self.related_args = []
        self.arg_map = {
            name : dtype for name, dtype in zip(arg_names, dtypes)
        }
        self.constr_flags : Dict[str, Dict[str, bool]] = {}
        self.other_flags : Dict[str, Dict[str, bool]] = {}
        self.min_len : Dict[str, int] = {}
    
    def is_in_argnames(self, arg, arg_names = None) :
        if arg_names is None : arg_names = self.arg_names 
        return arg in arg_names
    def gen_z3_obj(self, arg, arg_map, ret_wrapper=True, no_const=False) : 
        """
        in most of the case, 
        we need to return wrapper, 
        """
        
        if self.is_sym(arg) or is_wrapper(arg) : 
            z3obj = arg
        elif self.is_in_argnames(arg) :
            z3obj = arg_map[arg]
        else :
            if isinstance(arg, str) and no_const : # index 'i' generator
                return z3.Int(arg) 
            return arg # constant
        if is_wrapper(z3obj) and not ret_wrapper :
            # array but need to its const, why?
            return z3obj.get_wrapped_object()
        else :
            return z3obj

    def gen_exp_constr(self, generator, arg_map):
        # Assuming generator is a dictionary with keys "element" and "generators"
        constraints = []
        for comp in generator["generators"]:
            # Assuming comp is a dictionary with keys "target", "iter", "ifs"
            target = self.gen_z3_obj(comp["target"], arg_map, no_const=True)
            iter = self.gen_z3_obj(comp["iter"], arg_map, ret_wrapper=True)
            ifs = [self.gen_z3_obj(if_clause, arg_map) for if_clause in comp["ifs"]]
            # Check if iteration is over a list/array or a range
            if isinstance(iter, (ArrWrapper, TensorWrapper, z3.ArrayRef)) :
                is_z3_array = isinstance(iter, z3.ArrayRef)
                range = iter.range()
                lower_bound = range[0] if not is_z3_array else 0
                upper_bound = range[1] if not is_z3_array else z3funcs.len(iter)
                step = 1
                if ifs :
                    ifs = change_val_from_expr(ifs, target, iter[target])
                generator["element"] = change_val_from_expr(generator["element"], target, iter[target])
            elif isinstance(iter, z3.ArithRef) : 
                raise IncompatiableConstrError(f"{iter} should be Array, but got Int")
            else:
                # Case: Iteration over a range
                lower_bound, upper_bound, step = iter
            assert abs(step) < 2, f"step should be less than 2, cur : {step}"
            ### list comprehension
            is_in = z3.And(target >= lower_bound, target < upper_bound)
            # Use z3.And to combine all conditions in 'ifs'
            combined_ifs = z3.And(*ifs) if ifs else z3.BoolVal(True)

            # Element of the generator expression
            element = self.gen_z3_obj(generator["element"], arg_map)

            # Construct the Z3 constraint using z3.Imply
            constraint = z3.ForAll([target], z3.Implies(z3.And(is_in, combined_ifs), element))
            constraints.append(constraint)

        # Combine all constraints
        return constraints
    def update_related_args(self, related_args) :
        self.related_args.extend(related_args)
    def gen_empty_constr_flags(self, arg_names) :
        return {
            name : {
                flag_name : False for flag_name in FLAGS
            } for name in arg_names
        }
    def dtype_hijack(self, dtype) : 
        ## if dtype-related rule exist, then hijack dtype to be the same as the rule
        pass 
    def gen_dtype_obj(self) :
        return dict_combinations(self.arg_map) 

    def ret_dtype_var(self, info=None, min_len=None):
        """
        Returns a dictionary where each key is a variable name and the value is a list of 'True' flags and 'min_len'.
        """
        if info is None: info = self.constr_flags
        if min_len is None: min_len = self.min_len

        output_dict = {}
        for key, inner_dict in info.items():
            true_keys = [inner_key for inner_key, value in inner_dict.items() if value]
            output_list = true_keys.copy()
            if output_list:
                output_dict[key] = [z3.Bool(key + "_" + flag) for flag in output_list]

        return output_dict
    
    def gen_dtype_suff_conds(self, info=None) : 
        if info is None: info = self.constr_flags
        output_list = []
        for key, inner_dict in info.items():
            true_keys = [inner_key for inner_key, value in inner_dict.items() if value]
            if true_keys:
                output_list.append(z3.Or(*[z3.Bool(key + "_" + flag) for flag in true_keys]))
        return output_list
    
    def pretty_flags(self, info = None, min_len = None) :
        pretty_output = ""
        if info is None : info = self.constr_flags
        if min_len is None : min_len = self.min_len
        for key, inner_dict in info.items():
            true_keys = [inner_key for inner_key, value in inner_dict.items() if value]
            min_len_value = min_len.get(key, 0)

            if true_keys or min_len_value is not None:
                pretty_output += f"{key}: "
                if true_keys:
                    pretty_output += f"{', '.join(true_keys)}"
                if min_len_value is not None:
                    if true_keys:  # Add a separator if there are also true keys
                        pretty_output += ", "
                    pretty_output += f"min_len: {min_len_value}"
                pretty_output += "\n"
        return pretty_output.strip()  
    
    def parse(self, txt) : 
        try :
            ast_tree = ast.parse(txt, mode='eval')
            return ast_tree.body
        except :
            raise ValueError(f"Unsupported AST node {ast.dump(txt)})")
    def set_flag(self, name, container = None, **kwargs) :
        #must_iter=None, must_int=None, must_not_iter=None, must_str= 
        if self.is_in_argnames(name, self.related_args) : 
            if container is None : 
                container = self.constr_flags
            for key, value in kwargs.items() :
                container[name].update({key : value})

    def load_ast_with_hooks(self, txt, arg_names) : 
        """
        parse the AST from the txt.
        identify the name in the txt with given arg_names
        """
        cleaned_txt = clean_txt(txt)
        ast_tree = self.parse(cleaned_txt)
        related_args = identify_related_args(ast_tree, arg_names)
        self.update_related_args(related_args)
        return ast_tree
    
    def gen_len_suff_cond(self, z3objs : Dict[str, z3.Var]) -> z3.ExprRef : 
        results = []
        for name, flag in self.min_len.items() :
            if flag is not None :
                z3obj = z3objs[name].get_wrapped_object() if is_wrapper(z3objs[name]) else z3objs[name]
                results.append(z3.And(
                    self.min_len[name] >= 0,
                    self.min_len[name] < self.len(z3obj)
                ))
        if results : return z3.And(results)
        else : return 

    def conn_suff_conds(self, constr, z3_type_objs) -> z3.ExprRef : 

        def inspect_suff_conds(z3objs : List[z3.Var], body : z3.ExprRef) :
            for name, flag in self.constr_flags.items() :
                z3obj = z3objs[name].get_wrapped_object() if is_wrapper(z3objs[name]) else z3objs[name]
                if flag["must_iter"] and not self.is_iterable(z3obj) : return True 
                elif flag["must_not_iter"] and self.is_iterable(z3obj) : return True
                elif flag["must_int"] and not self.is_int(z3obj) : return True
                elif flag["must_str"] and not self.is_str(z3obj) : return True

            return body
        
        len_suff_con = self.gen_len_suff_cond(z3_type_objs)
        if len_suff_con is not None :
            constr = z3.Implies(len_suff_con, constr)

        return partial(inspect_suff_conds, body = constr)

    def gen_func_obj(self, func_name, *args) :
        if all(isinstance(arg, str) for arg in args) :
            constants = ".".join(list(args)+[func_name]) # tf.complex12 -> ast identifies it as attributes
            if is_dtype_constant(constants) :
                return get_dtype_z3_obj(constants)
        self.set_flag_by_func_name(func_name, args)
        if not hasattr(self, func_name) : 
            raise IncorrectConstrError(f"func name : \"{func_name}\" is not implemented.")
        func_exec = getattr(self, func_name)
        args = self.get_syms_from_wrappers(*args)
        res = func_exec(*args)
        return res

    def set_ele_type_flag(self, name, comparator) : 
        if isinstance(comparator, list) :
            comparator = comparator[0]
        if is_wrapper(comparator) : 
            if self.is_int(comparator.get_wrapped_object()) : self.set_flag(name, must_num=True)
            if self.is_str(comparator.get_wrapped_object()) : self.set_flag(name, must_str=True)
        elif self.is_sym(comparator) : 
            if self.is_int(comparator) : self.set_flag(name, must_num=True)
            if self.is_str(comparator) : self.set_flag(name, must_str=True)
        else : 
            if isinstance(comparator, int) : self.set_flag(name, must_num=True)
            if isinstance(comparator, str) : self.set_flag(name, must_str=True)   

    def set_min_len(self, name, idx) : 
        if isinstance(idx, int) :      
            if name in self.min_len.keys() : 
                if self.min_len[name] is None :
                    self.min_len[name] = idx
                else :
                    self.min_len[name] = max(self.min_len[name], idx)

    def set_flag_by_func_name(self, func_name, args) :
        if func_name in iter_specific_funcs :
            for arg in args : 
                if self.is_sym(arg) or is_wrapper(arg) :
                    self.set_flag(self.get_name(arg), must_iter=True)
        if func_name in tensor_dtype_check_funcs : 
            for arg in args : 
                if self.is_sym(arg) or is_wrapper(arg) :
                    self.set_flag(self.get_name(arg), 
                                  container=self.other_flags,
                                  is_tensor_dtype=True)
    @staticmethod
    def gen_sym(a):
        # Check if 'a' is a Z3 expression or sort
        return z3.Int(a)


    def gen_basic_constr(self, op, *args) : 
        if len(args) > 1 : 
            if not all(
                self.is_sym(arg) or is_wrapper(arg) for arg in args
            ) : 
                args = list(args)
                args[0] = self.gen_z3_obj(args[0], self.arg_map, no_const=True, ret_wrapper = False)  
            
            left_name = self.get_name(args[0]) if self.is_sym(args[0]) or is_wrapper(args[0]) else args[0]
            right_name = self.get_name(args[1]) if self.is_sym(args[1]) or is_wrapper(args[1]) else args[1]
            if left_name in self.related_args and not self.is_func_applied(args[0]) :
                self.set_ele_type_flag(left_name, args[1])
            if right_name in self.related_args and not self.is_func_applied(args[1]) :
                self.set_ele_type_flag(left_name, args[0])
        # set flag( len(args)> 2, not in, and not_in, make operation comparable )
        try :
            res = get_operator(op)(*args)
        except :
            raise IncompatiableConstrError(f"INCOMPATIABLE : {op}, {args}")
        return res
    
    def gen_sliced_obj(self, obj, arg_map, start = None, end = None) :
        z3obj = self.gen_z3_obj(obj, arg_map, ret_wrapper=True)
        if start is None : start = 0 
        if end is None : end = z3obj.rank
        idx_range = (z3obj.corrected_idx(start), z3obj.corrected_idx(end))
        self.set_min_len
        z3obj.update_info(sliced=idx_range)
        return z3obj
    
    def gen_bool_constr(self, op, *args) : 
        res = get_bool_operator(op)(*flatten_list(args))
        return res

    def make_compatiable(self, left, op, right) : 
        new_right = right
        new_op = op
        if isinstance(right, list) : 
            new_right = list(set(flatten_list(right)))
            new_op = self.replace_op_accord_constant(op, new_right)
            return new_op, new_right
        if is_dtype_constant(right) : # TODO : very inefficient(every conversion need to check)
            name = self.get_name(left)
            new_right = get_dtype_z3_obj(right, 
                                    is_tensor_dtype=self.other_flags[name].get("is_tensor_dtype", False)
            )
            new_op = self.replace_op_accord_constant(op, new_right)
        return new_op, new_right
    
    def replace_op_accord_constant(self, op, arg) : 
        """
        some dtype is float -> float16, float32, float64
        eq -> in 
        not_eq -> not in 
        """
        if hasattr(arg, "__len__") : # suppose that arg is replaced
            if is_same_ast_name(op, ast.Eq) : 
                return ast.In.__name__
            elif is_same_ast_name(op, ast.NotEq) :
                return ast.NotIn.__name__
        return op

    
    def convert(self) : 
        result = None 
        ast = self.load_ast_with_hooks(self.txt, self.arg_names)
        self.constr_flags = self.gen_empty_constr_flags(self.related_args)
        self.other_flags = {name : {} for name in self.related_args}
        self.min_len = {name : None for name in self.related_args}
        for dtype_map in self.gen_dtype_obj() : 
            # dtype_hijack else : 
            # if hasattr(dtype_map.get('dilation_rate',0), "arg_type") :
            #     print(dtype_map
            # )
            z3_type_objs = {
                    name : dtype.z3()(name) for name, dtype in dtype_map.items()
                }
            try : 
                result = self._convert(ast, z3_type_objs)
                if result is not None :
                    result = self.conn_suff_conds(result, z3_type_objs)        
                break
            except IncompatiableConstrError :
                continue
            # except :
            #     raise ValueError(f"Unexpected error : {traceback.format_exc()}")
        return result
    def gen_z3_obj_from_all_defined_field(self, val, arg_map, ret_wrapper=True, no_const=False) :
        if not self.is_in_argnames(val) and is_dtype_constant(val) : # TODO : very inefficient(every conversion need to check)
            return get_dtype_z3_obj(val)
        else : 
            return self.gen_z3_obj(
                            val,
                            arg_map,
                            ret_wrapper=ret_wrapper,
                            no_const=no_const
                            )
    def _convert(self, node, arg_map):
        if isinstance(node, ast.BoolOp):
            op = type(node.op).__name__
            values = [self._convert(value, arg_map) for value in node.values]
            return self.gen_bool_constr(op, *values)
        elif isinstance(node, ast.UnaryOp):
            op = type(node.op).__name__
            operand = self._convert(node.operand, arg_map)
            if is_same_ast_name(op, ast.Not) :
                return self.gen_bool_constr(op, operand)
            else : 
                return self.gen_basic_constr(op, operand)
        elif isinstance(node, ast.Compare):
            results = []
            left = self._convert(node.left, arg_map)
            for op, right_node in zip(node.ops, node.comparators):
                right = self._convert(right_node, arg_map)
                op_type = type(op).__name__
                op_type, right = self.make_compatiable(left, op_type, right)
                results.append(self.gen_basic_constr(op_type, left, right))
            return merge_constr(results)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attribute_value = self._convert(node.func.value, arg_map)
                return self.gen_func_obj(node.func.attr, attribute_value)
            else:
                func_name = node.func.id
                args = [self._convert(arg, arg_map) for arg in node.args]
                assert func_name in z3funcs.function_names, f"Unsupported function {func_name}"
                return self.gen_func_obj(func_name, *args)
            
        elif isinstance(node, ast.Attribute):
            value = self._convert(node.value, arg_map)
            return self.gen_func_obj(node.attr, value)
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op).__name__
            left = self._convert(node.left, arg_map)
            right =self._convert(node.right, arg_map)
            return self.gen_basic_constr(op_type, left, right)
        elif isinstance(node, ast.Subscript):
            # Handle negative indices and slicing
            if isinstance(node.slice, ast.Index):
                slice_value = self._convert(node.slice.value, arg_map)
                slice_value = self.gen_z3_obj(slice_value, arg_map, no_const=True)
                obj = self._convert(node.value, arg_map)
                obj_name = self.get_name(obj)
                self.set_min_len(obj_name, slice_value)
                self.set_flag(obj_name, must_iter=True)
                val = self.gen_z3_obj(obj, arg_map, ret_wrapper=True)
                slice_value = self.get_corrected_idx(slice_value, val)
                return val[slice_value]
            elif isinstance(node.slice, ast.Slice):
                # Slicing, e.g., a[1:] or a[:-1]
                start = self._convert(node.slice.lower, arg_map) if node.slice.lower else None
                end = self._convert(node.slice.upper, arg_map) if node.slice.upper else None
                array = self._convert(node.value, arg_map)
                return self.gen_sliced_obj(array, arg_map, start, end)

        elif isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            # Handle generator expressions
            elt = self._convert(node.elt, arg_map)
            generators = [self._convert(gen, arg_map) for gen in node.generators]
            generator_exp = {"type": "GeneratorExp", "element": elt, "generators": generators}
            return self.gen_exp_constr(generator_exp, arg_map)
        elif isinstance(node, ast.comprehension):
            # Handle comprehension part of generator expression
            target = self._convert(node.target, arg_map)
            iter = self._convert(node.iter, arg_map)
            ifs = [self._convert(if_clause, arg_map) for if_clause in node.ifs]
            comprehension = {"type": "comprehension", "target": target, "iter": iter, "ifs": ifs}
            return comprehension
        elif isinstance(node, ast.IfExp):
            # Handle IfExp (Ternary Conditional Expression)
            test = self._convert(node.test, arg_map)
            body = self._convert(node.body, arg_map)
            orelse = self._convert(node.orelse, arg_map)
            return f"({body} if {test} else {orelse})"
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [self._convert(elem, arg_map) for elem in node.elts]
        elif isinstance(node, ast.Name):
            return self.gen_z3_obj_from_all_defined_field(node.id, arg_map, ret_wrapper=True)
        elif isinstance(node, ast.Constant):
            return self.gen_z3_obj_from_all_defined_field(node.value, arg_map, ret_wrapper=True)
        elif isinstance(node, ast.Num):
            return node.n
        else:
            raise ValueError(f"Unsupported AST node {ast.dump(node)})")
            
    # Process and print each constraint
