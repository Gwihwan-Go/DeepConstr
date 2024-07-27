
## collect the number of constr for each operator
import os
import yaml
from matplotlib import pyplot as plt
import statistics
from typing import List
# -> visualize distribution of length of cosntr, f1_ prec_ recall of constr, pie chart of constr type

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=23)  # fontsize of the tick labels
plt.rc("ytick", labelsize=23)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIGGER_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC = 0.85
MAX_FAC = 1.02

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{xfrac}")
colors = ['blue', 'red']
plt.rcParams['text.usetex'] = True
# Define the path where to search for YAML file

def load_data(path) :
    data_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file ends with .yaml
            if file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                # Open and parse the YAML file
                with open(file_path, 'r') as stream:
                    try:
                        data = yaml.safe_load(stream)
                        if data.get("rules", None) is not None : # data already trained
                            data_list.append(data)
                    except yaml.YAMLError as exc:
                        print(exc)
    return data_list

def count_sub_constr(constr) :
    num = 1
    txt = constr["txt"]
    and_count = txt.count("and")
    or_count = txt.count("or")
    return num + and_count + or_count

def get_deepconstr_stats(data_list):
    deepconstr_stats = {
        "processed": 0,
        "LLM": 0
    }
    deepconstr_len = []
    deepconstr_operator = {}
    deepconstr_f1 = []
    deepconstr_prec = []
    deepconstr_recall = []
    deepconstr_rule_num = []
    api_per_subconstraints = {}
    overall_nsubconstr = 0
    for data in data_list:
        # if any([rule[0]["cot"] in ["processed", "divided"] for rule in data["rules"]]):
        #     deepconstr_stats["processed"] += 1
        # else :
        #     # for r in data["rules"] :
        #     #     print(r[0]["cot"])
        #     deepconstr_stats["LLM"] += 1
        if data["name"] not in deepconstr_operator:
            api_per_subconstraints[data["name"]] = 0
        for rule in data["rules"]:
            num_of_sub_constr = count_sub_constr(rule[0])
            if "processed" == rule[0]["cot"] or "divided" == rule[0]["cot"]:
                deepconstr_stats["processed"] += 1
            else :
                deepconstr_stats["LLM"] += 1
            api_per_subconstraints[data["name"]] += num_of_sub_constr
            if rule[0]["cot"] == "default" :
                continue
            deepconstr_len.append(rule[0].get("length", 1))
            deepconstr_f1.append(rule[1]["f1_score"])
            deepconstr_prec.append(rule[1]["precision"])
            deepconstr_recall.append(rule[1]["recall"])
    
    total = sum(api_per_subconstraints.values())
    print(" ## Num of Sub Constraints : ", total, "from", len(list(api_per_subconstraints.keys())), "number of operators")
    print(" ## Mean : ", total/len(list(api_per_subconstraints.keys())), " Median ",  statistics.median(api_per_subconstraints.values()))
    return deepconstr_stats, deepconstr_len, deepconstr_operator, deepconstr_f1, deepconstr_prec, deepconstr_recall

def mean(numbers):
    return sum(numbers) / len(numbers)

def viz_passrate(data_list, deepdeepconstr_s_data_list, name, path = "/DeepConstr/results/") :
    all = []
    for i, data_li in enumerate([data_list, deepdeepconstr_s_data_list]):
        pass_rate_num_of_constr = []
        for data in data_li:
            pass_rate_num_of_constr.append(
                (data["pass_rate"], len(data["rules"]))
            )
        pass_rate, num_of_constraints = zip(*pass_rate_num_of_constr)
        all.extend(pass_rate)
        print("mean passrate", mean(pass_rate))
        print("median passrate", statistics.median(pass_rate))

    print("overall passrate")
    print("mean passrate", mean(all))
    print("median passrate", statistics.median(all))

def viz_gen_way_of_constrs(deepconstr_stats):
    print("### GEN WAY OF CONSTRAINS ###")
    print(deepconstr_stats)
    print("### GEN WAY OF CONSTRAINS ###")

def viz_deepconstr_len(deepconstr_len : List[int]):
    plt.hist(deepconstr_len, bins=range(1, max(deepconstr_len) + 1))
    plt.show()

def viz_f1(deepconstr_recall : List[float], deepconstr_prec, deepdeepconstr_s_recall, deepdeepconstr_s_prec , path = "/DeepConstr/results/", name = "PyTorch"):

    plt.figure(figsize=(9,9))
    plt.legend(loc='lower right')
    # PyTorch
    plt.scatter(deepdeepconstr_s_prec, deepdeepconstr_s_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr$^{s}$}', linewidth=0.6, s=80, c='blue', marker='x')
    plt.scatter(deepconstr_prec, deepconstr_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr}', linewidth=0.6, s=80, c='red', marker='+')
    # plt.scatter(deepdeepconstr_s_prec, deepdeepconstr_s_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr$^{s}$}', linewidth=0.6, s=80, c='blue', marker='o')
    # plt.scatter(deepconstr_prec, deepconstr_recall, alpha=0.8, edgecolors='black', label=r'\textsc{DeepConstr}', linewidth=0.6, s=80, c='red', marker='o')
    print("mean deepdeepconstr^s_soundness", mean(deepdeepconstr_s_prec))
    print("median deepdeepconstr^s_soundness", statistics.median(deepdeepconstr_s_prec))
    print("mean deepdeepconstr^s_completeness", mean(deepdeepconstr_s_recall))
    print("median deepdeepconstr^s_completeness", statistics.median(deepdeepconstr_s_recall))
    print("mean deepconstr_soundness", mean(deepconstr_prec))
    print("median deepconstr_soundness", statistics.median(deepconstr_prec))
    print("mean deepconstr_completeness", mean(deepconstr_recall))
    print("median deepconstr_completeness", statistics.median(deepconstr_recall))
    # if name == "torch" :
    #     plt.title('\\textit{PyTorch}')
    # else :
    #     plt.title('\\textit{TensorFlow}')

    plt.xlabel('\\textit{Soundness}', fontsize=20)
    plt.ylabel('\\textit{Completeness}', fontsize=20)
    plt.legend()
    plt.xticks(range(0, 101, 10))
    plt.yticks(range(0, 101, 10))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if os.path.exists(path) == False:
        os.makedirs(path)
    plt.savefig(path + f"5_dist_{name}.pdf")
    plt.savefig(path + f"5_dist_{name}.png")

if __name__ == "__main__" : 
    record_dir = "/DeepConstr/data/"
    frameworks = ["torch", "tf"] #, "numpy"]
    kinds = ["records", "only_acc"]

    for framework in frameworks:
        print("####### ", framework, " ####### ")
        data = []
        path = os.path.join(record_dir, "records", framework)
        deepdeepconstr_s_path = os.path.join(record_dir, "only_acc", framework)
        data_list = load_data(path)
        
        deepdeepconstr_s_data_list = load_data(deepdeepconstr_s_path)
        print("DeepConstr")
        deepconstr_stats, deepconstr_len, deepconstr_operator, deepconstr_f1, deepconstr_prec, deepconstr_recall = get_deepconstr_stats(data_list)
        print("DeepConstr^s")
        deepdeepconstr_s_deepconstr_stats, deepdeepconstr_s_deepconstr_len, deepdeepconstr_s_deepconstr_operator, deepdeepconstr_s_deepconstr_f1, deepdeepconstr_s_deepconstr_prec, deepdeepconstr_s_deepconstr_recall = get_deepconstr_stats(deepdeepconstr_s_data_list)

        viz_f1(deepconstr_recall, deepconstr_prec, deepdeepconstr_s_deepconstr_recall, deepdeepconstr_s_deepconstr_prec, name=framework)
    # for framework in frameworks:
    #     data = []
    #     path = os.path.join(record_dir, "records", framework)
    #     deepdeepconstr_s_path = os.path.join(record_dir, "only_acc", framework)
    #     data_list = load_data(path)
    #     deepdeepconstr_s_data_list = load_data(deepdeepconstr_s_path)
    #     print("DeepConstr")
    #     deepconstr_stats, deepconstr_len, deepconstr_operator, deepconstr_f1, deepconstr_prec, deepconstr_recall = get_deepconstr_stats(data_list)
    #     print("DeepConstr^s")
    #     deepdeepconstr_s_deepconstr_stats, deepdeepconstr_s_deepconstr_len, deepdeepconstr_s_deepconstr_operator, deepdeepconstr_s_deepconstr_f1, deepdeepconstr_s_deepconstr_prec, deepdeepconstr_s_deepconstr_recall = get_deepconstr_stats(deepdeepconstr_s_data_list)
        # print("all")
        # viz_gen_way_of_constrs(deepconstr_stats)
        # print("only_acc")
        # viz_gen_way_of_constrs(deepdeepconstr_s_deepconstr_stats)
        # viz_passrate(data_list, deepdeepconstr_s_data_list, name=framework)
        # print(pass_rate_num_of_constr)
