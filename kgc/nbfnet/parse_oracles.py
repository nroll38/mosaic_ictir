import json
import copy
import numpy as np
import csv

from scipy import stats

import matplotlib.pyplot as plt

#secret->removed triple -> mrr
def ranked_list(data):
    out = {}
    lookup = {}
    for secret, removal_dict in data.items():
        out[secret] = []
        lookup[secret] = {}
        if "original" in removal_dict:
            base_score = removal_dict["original"]
        #print("BASE: {}".format(base_score))
        for triple, mrr in removal_dict.items():
            #print(triple)
            #print("NEW: {} \t(BASE: {}".format(mrr, base_score))
            if triple == "original":
                continue
            out[secret].append((triple, base_score - mrr))
            lookup[secret][triple] = max(base_score - mrr, 0)
            
        out[secret].sort(key=lambda x: x[1], reverse=True)
        #if out[secret] == []:
            

    return out, lookup

def rrf(l1, l2):
    scores = {}
    out = []
    for r1, (secret, score) in enumerate(l1):
        for r2, (secret2, score2) in enumerate(l2):
            if secret == secret2:
                scores[secret] = 1/(60+r1) + 1/(60+r2)
        if secret not in scores:
            scores[secret] = 1/(60+r1+1) + 1/(60+len(l2)+1)
            

    for r2, (secret2, score2) in enumerate(l2):
        for r1, (secret, score) in enumerate(l1):
            if secret == secret2:
                scores[secret] = 1/(60+r1) + 1/(60+r2)
        if secret2 not in scores:
            scores[secret2] = 1/(60+len(l1)+1) + 1/(60+r2+1)
            
    out = list(scores.items())
    out.sort(key=lambda x: x[1], reverse=True)
    #print(out)
    return out
    
def get_ensemble(model1, model2):
    result = {}

    for secret in model1:
        #print(len(model2[secret]))
        if secret in model2:
            result[secret] = rrf(model1[secret], model2[secret])
        else:
            result[secret] = model1[secret]
    for secret in model2:
        if secret not in result:
            result[secret] = model2[secret]

    return result



def get_lists(fp):
    with open(fp) as fd:
        data = json.load(fd)

    
    return ranked_list(data)

def get_centrality_divisions(centrality_stats, splits=1/3):
    center_max = []
    center_median = []

    #print(len(centrality_stats))

    for secret, (s_max, s_med) in centrality_stats.items():
        center_max.append((secret, s_max))
        center_median.append((secret, s_med))

    center_max.sort(key=lambda x: x[1], reverse=True)
    center_median.sort(key=lambda x: x[1], reverse=True)

    max_splits = []
    med_splits = []
    split = splits
    while split < 1:
        next_split = center_max[int(split * len(center_max))]
        max_splits.append(next_split[1])

        next_split = center_median[int(split * len(center_median))]
        med_splits.append(next_split[1])
        
        split += splits

    max_splits.append(center_max[-1][1]-1)
    med_splits.append(center_median[-1][1]-1)

    max_dict = {}
    med_dict = {}

    for secret, (s_max, s_med) in centrality_stats.items():
        found = False
        for i, split in enumerate(max_splits):
            if s_max > split:
                max_dict[str(secret)] = i
                found = True
                break
            #if not found:
                
        for i, split in enumerate(med_splits):
            if s_med > split:
                med_dict[str(secret)] = i
                break
        
                
    #print(len(med_dict))

    #print(max_dict.values())
        
    return max_dict, med_dict#max_splits, med_splits

def mrr_splits(split_dict, secrets, score_list, result_eval):

    splits = set()
    split_lists = []
    result_lists = []
    out_list = []
    for val in split_dict.values():
        splits.add(val)
    #print(len(split_dict.keys()))

    for i in range(len(splits)):
        split_lists.append([])
        result_lists.append([])

    #print(split_lists)
    for i, secret in enumerate(secrets):
        #print(i)
        if secret not in split_dict:
            continue
        idx = split_dict[secret]
        #print(idx)
        split_lists[idx].append(score_list[i])
        result_lists[idx].append(result_eval[i])

    #print(result_lists)
    for i in range(len(split_lists)):
        #print(np.mean(split_lists[i]))
        #print(np.mean(result_lists[i]))
        # exit()
        
        out_list.append(abs(np.mean(split_lists[i])/np.mean(result_lists[i])))
                                
        
    #print(out_list)
    return(out_list)


def get_eval_divisions(results, evaluator, secrets):

    secret_splits = []

    secret_dict = {}

    secret_lists = [[], [], []]

    print(np.count_nonzero(results[evaluator]))

    eval_tuples = list(zip(results[evaluator], list(range(0, len(results[evaluator])))))
    eval_tuples.sort(key=lambda x: x[0], reverse=True)    
    #print(eval_tuples)

    for i, entry in enumerate(eval_tuples):
        if entry[0] == 0:
            break
    eval_tuples = eval_tuples[:i]
    #print(eval_tuples)
    
    print(np.mean([i for i, j in eval_tuples]))

    split = len(eval_tuples)/3

    for i, (_, secret_idx) in enumerate(eval_tuples):
        #secret_dict[secrets[secret_idx]] = int(i/split)
        secret_dict[secret_idx] = int(i/split)

        secret_lists[int(i/split)].append(secret_idx)
        #print(int(i/split))
    
    
    #for entry in results[evaluator]:

    #exit()    
    return secret_dict, secret_lists


def get_mrr_stats(split_dict, secrets, score_list, result_eval):
    splits = set()
    split_lists = []
    result_lists = []
    out_list = []
    for val in split_dict.values():
        splits.add(val)
    #print(len(split_dict.keys()))

    for i in range(len(splits)):
        split_lists.append([])
        result_lists.append([])

    #print(split_lists)
    for i, secret in enumerate(secrets):
        #print(i)
        if secret not in split_dict:
            continue
        idx = split_dict[secret]
        #print(idx)
        split_lists[idx].append(score_list[i])
        result_lists[idx].append(result_eval[i])

    #print(result_lists)
    for i in range(len(split_lists)):
        #print(np.mean(split_lists[i]))
        #print(np.mean(result_lists[i]))
        # exit()
        
        out_list.append(abs(np.mean(split_lists[i])/np.mean(result_lists[i])))
                                
        
    #print(out_list)
    return(out_list)
    
def create_plots(results):

    base = ("RuleN", "GraIL", "NBFNet")

    lists = [results["rulen"]["rulen"],
             results["grail"]["grail"],
             results["nbf"]["nbf"],]
    
    # for k, result in results.items():
    #     lists_og.append(result[k])



    lists = list(zip(*lists))
    lists.sort(key=lambda x: x[0], reverse=True)
    
    to_graph = [[i for i, j, k in lists],
                [j for i, j, k in lists],
                [k for i, j, k in lists]
                ]

    #print(to_graph[2])

    for i, y in enumerate(to_graph):
        x = list(range(0, len(y)))
        
        plt.bar(x, y, label="MRR Change By Secret: " + base[i], color="black")
        plt.xlabel("Secrets")
        plt.ylabel("MRR Change")
        plt.ylim(0, 1.0)
        plt.savefig("./mrr_change_plot_" + base[i].lower()+".png")
        plt.close()
        
if __name__ == "__main__":

    rulen_fp = "experiments/exp_v1_33/rulen_oracle_results.json"
    grail_fp = "experiments/exp_v1_33/grail_oracle_results.json"
    nbf_fp = "../NBFNet/oracle_results.json"
    centrality_fp = "experiments/exp_v1_33/centrality.json"
    output_dp = "experiments/exp_v1_33/"

    # rulen_fp = "experiments/exp_v1_33/rulen_oracle_results.json"
    # grail_fp = "experiments/exp_v1_33/grail_oracle_results.json"
    # nbf_fp = "../NBFNet/oracle_results.json"
    # centrality_fp = "experiments/exp_v1_33/centrality.json"
    # output_dp = "experiments/exp_v1_33/"

    models = {"rulen": [],
              "grail": [],
              "nbf": [],
              "r+g": [],
              "r+n": [],
              "g+n": []
              }

    idx = {"rulen": 0,
              "grail": 1,
              "nbf": 2,
              "r+g": 3,
              "r+n": 4,
              "g+n": 5
              }
    lookup_scores = {}

    with open(centrality_fp, "r") as fd:
        centrality_stats = json.load(fd)

    #max_splits, med_splits = get_centrality_divisions(centrality_stats)
    
    results = {"rulen": copy.deepcopy(models),
               "grail": copy.deepcopy(models),
               "nbf": copy.deepcopy(models),}

    #(models["grail"], lookup_scores["grail"])  = get_lists(grail_fp)
    #exit()

    (models["rulen"], lookup_scores["rulen"]) = get_lists(rulen_fp)
    (models["grail"], lookup_scores["grail"])  = get_lists(grail_fp)
    (models["nbf"], lookup_scores["nbf"]) = get_lists(nbf_fp)
    models["r+g"] = get_ensemble(models["rulen"], models["grail"])
    models["r+n"] = get_ensemble(models["rulen"], models["nbf"])
    models["g+n"] = get_ensemble(models["grail"], models["nbf"])

    #get_ensemble(models["rulen"], modes["nbf"])


    print(len(lookup_scores["rulen"].keys()))
    print(len(lookup_scores["grail"].keys()))
    print(len(lookup_scores["nbf"].keys()))

    secrets = list(models["rulen"].keys())

    for secret in secrets:
        for evaluator, result in results.items():
            
            for model, ranks in models.items():
                if secret not in ranks or not ranks[secret]:
                    result[model].append(0)
                    continue
                nomination = ranks[secret][0][0]
                removal_score = 0
                #print("EVAL: {}".format(evaluator))
                #print("Secret: {}".format(secret))
                #print("Nomination: {}".format(nomination))
                if nomination in lookup_scores[evaluator][secret]:
                    removal_score = lookup_scores[evaluator][secret][nomination]
                #print(removal_score)
                result[model].append(removal_score)
            
  
    create_plots(results)
        
    total_eval_set = set()
    for evaluator, result in results.items():
        print("EVAL: {}".format(evaluator))

        eval_dicts, eval_lists = get_eval_divisions(results[evaluator], evaluator, secrets)

        total_eval_set.update(eval_dicts.keys())
        #print(eval_lists)
        
        
        output_fp = output_dp + evaluator + ".csv"
        with open(output_fp, "w") as fd:
            writ = csv.writer(fd)
            sig_check = [result[evaluator]]
            for model, score_list in result.items():
                output = [model]
                output.append(abs(np.mean(score_list)/np.mean(result[evaluator])))
                #output += score_centrality(max_splits, secrets, score_list, result[evaluator])
                #output += score_centrality(med_splits, secrets, score_list, result[evaluator])
                
                print(output)
                if model == "rulen" and (evaluator == "grail"):
                    t_stat, p_value = stats.ttest_rel(score_list, result["nbf"])
                    sig_check.append(score_list)

                    #print(score_list)
                    #print(result[evalautor])
                    print("MODEL: {}\t Other: {}\t P: {}".format(model,"nbf", p_value))

                if model == "rulen" and (evaluator == "nbf"):
                    t_stat, p_value = stats.ttest_rel(score_list, result["grail"])
                    sig_check.append(score_list)

                    #print(score_list)
                    #print(result[evalautor])
                    print("MODEL: {}\t Other: {}\t P: {}".format(model,"grail", p_value))
                    

                if model == "grail" and (evaluator == "rulen"):
                    t_stat, p_value = stats.ttest_rel(score_list, result["nbf"])
                    sig_check.append(score_list)

                    #print(score_list)
                    #print(result[evalautor])
                    print("MODEL: {}\t EVAL: {}\t P: {}".format(model, "nbf", p_value))
                writ.writerow(output)

                
        output_fp = output_dp + evaluator + "_eval_splits.csv"
        with open(output_fp, "w") as fd:
            writ = csv.writer(fd)
            sig_check = [result[evaluator]]

            model_idx = {}
            output = []
            for i, model in enumerate(result.keys()):
                output.append("")
                output.append("")
                model_idx[model] = i

            for model, i in model_idx.items():
                output[i] = model
                output[i + len(result.keys())] = model
            writ.writerow(output)

            print(eval_lists)
            for i, eval_list in enumerate(eval_lists):
                eval_results_dict = {}
                for model in result.keys():
                    eval_results_dict[model] = []

                for secret_idx in eval_list:
                    for model, result_list in result.items():
                        eval_results_dict[model].append(result_list[secret_idx])


                output = []
                for i, model in enumerate(result.keys()):
                    output.append("")
                    output.append("")

                    
                output[model_idx[evaluator]] = np.mean(eval_results_dict[evaluator])
                output[model_idx[evaluator] + len(result.keys())] = np.mean(eval_results_dict[evaluator])
                print(np.mean(eval_results_dict[evaluator]))

                for model, result_list in eval_results_dict.items():
                    if model == evaluator:
                        continue
                    
                    output[model_idx[model]] = np.count_nonzero(result_list)/len(eval_list)
                    #print(np.count_nonzero(result_list)/len(eval_list))

                    output[model_idx[model] + len(result.keys())] = abs(np.mean(result_list)/np.mean(eval_results_dict[evaluator]))  #len(result_list)/len(eval_list)
                print(output)
                writ.writerow(output)
                #exit()    
                #np.mean(eval_results_dict[evaluator])

                                          
                                          
                        
            # t_stat, p_value = stats.f_oneway(*sig_check)
            # print("ANOVA P: {}".format(p_value))

                #print("Eval: {} \t Model: {}".format(evaluator, model))
                #print("Change: {}".format(abs(np.mean(score_list)/np.mean(result[evaluator]))))
                
            
                
        
            

    print("Total non-zero secrets: {}".format(len(total_eval_set)))
    print(total_eval_set)
    for evaluator, result in results.items():
        out = []
        for secret_index in total_eval_set:
            out.append(result[evaluator][secret_index])
        print("{}: {}".format(evaluator, np.mean(out)))
        #print(result)
