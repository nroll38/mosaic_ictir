

import numpy as np
from tqdm import tqdm
import csv
import random
import copy
import os
import json

def load_data(fp):
    data = {}
    data_s = {}
    data_write = []

    with open(fp, "r") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            (s, r, o)  = row
            data_write.append((s, r, o))
            
            if r not in data:
                data[r] = {}
            if s not in data[r]:
                data[r][s] = [o]
            else:
                data[r][s].append(o)

            if s not in data_s:
                data_s[s] = {}
            if r not in data_s[s]:
                data_s[s][r] = [o]
            else:
                data_s[s][r].append(o) 
    
    return data, data_s, data_write

def load_set(fp):

    ret_set = set()
    with open(fp, "r") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            (s, r, o)  = row
            ret_set.add((s,r,o))
    return ret_set
            
#given s, target, steps, find surrounding nodes and recusively call until target found or max steps(k) reached
def find_surrounding_nodes_triples(data_s, s, target_o, steps, k, paths, prev_path):
    if s not in data_s:
        return
    
    for r, o_list in data_s[s].items():
        for o in o_list:
            #print("PREV PATH: {}".format(len(prev_path)))
            cur_path = prev_path + (s,r)
            #print("CUR PATH: {}".format(len(cur_path)))
            #print("CUR PATH: {}".format(cur_path))
            if o == target_o:
                cur_path += (o,)
                path = tuple(cur_path)
                #print("PATH: {}".format(path))
                if path not in paths:
                    paths[path] = 1
                else:
                    paths[path] += 1
                continue
            if steps < k-1:
                find_surrounding_nodes_triples(data_s, o, target_o, steps + 1, k, paths, cur_path)
                

#extract the actual triples from paths of form n1 -> r1 -> n2 -> r2 -> n3
def get_triples_paths(paths, s, o):
    print("*"*20)
    print("S: {}\t O: {}".format(s,o))
    tripled_paths = set()
    for path in paths:
        
        for i in range(0, len(path)-1, 2):
            #print(path[i] + ", " + path [i+1] + ", " + path[i+2])
            tripled_paths.add((path[i], path[i+1], path[i+2]))

    return tripled_paths

def write_minus(fp, data_write, triple, style="w"):
    with open(fp, style) as fd:
        writ = csv.writer(fd, delimiter="\t")
        for row in data_write:
            if row != triple:
                writ.writerow(row)

#grabs RR from GraIL log
def parse_log(fp):
    with open(fp, "r") as fd:
        for line in fd.readlines():
            pass
            
    return float(line)
            
                
if __name__ == "__main__":
    data_fp = "./data/exp_v1_33_select/test/train_og.txt"
    secret_fp = "./data/exp_v1_33_select/secrets.txt"
    new_data_fp = "./data/exp_v1_33_select/new.txt"

    train_fp = "./data/exp_v1_33_select/test/train.txt"
    test_fp = "./data/exp_v1_33_select/test/test.txt"

    log_fp = "./rulen_mrr.txt"#"./experiments/exp_v1_33/log_rank_test.txt"

    results_fp = "experiments/exp_v1_33_select/rulen_oracle_results.json"


    # data_fp = "./data/exp_v1_33/test/train_og.txt"
    # secret_fp = "./data/exp_v1_33/secrets.txt"
    # new_data_fp = "./data/exp_v1_33/new.txt"

    # train_fp = "./data/exp_v1_33/test/train.txt"
    # test_fp = "./data/exp_v1_33/test/test.txt"

    # log_fp = "./rulen_mrr.txt"#"./experiments/exp_v1_33/log_rank_test.txt"

    # results_fp = "experiments/exp_v1_33/rulen_oracle_results.json"
    #new_secrets = "./data/exp_v1_33/secrets.txt"
    results = {}
    best_changes = []
    
    data, data_s, data_write = load_data(data_fp)
    secrets = load_set(secret_fp)
    new_data = load_set(new_data_fp)

    all_train_data = load_set(data_fp)

    #improve secrets:
    #count = 11
    for (s,r,o) in secrets:
        secret_triple = (s,r,o)
        paths = {}
        prev_path = ()
        triple_count = 0

        # if secret_triple in secrets:
        #     continue
        # write_minus(train_fp+"2", [secret_triple], "", "a")
        # continue
    
        # find_surrounding_nodes_triples(data_s, s, o, 0, 3, paths, prev_path)
        # path_triples = get_triples_paths(paths, s, o)

        # if len(path_triples) > 4:
        #     write_minus(new_secrets, [secret_triple], "", "a")
        #     count += 1
        # if count == 50:
        #     break
        # continue    
        
        results[str(secret_triple)] = {}
        #ensure test, train  files initialized to full graph and target secret; clear log
        write_minus(test_fp, [secret_triple], "")
        write_minus(train_fp, data_write, "")
        open(log_fp, "w").close()


        #get triples for each component of a path IOT ID if a valid new triple for potential break
        paths = {}
        prev_path = ()
        triple_count = 0
        try:
            find_surrounding_nodes_triples(data_s, s, o, 0, 3, paths, prev_path)
            path_triples = get_triples_paths(paths, s, o)
            os.system("python my_rulen.py")# -d exp_v1_33/test -e exp_v1_33")
            og_rank = parse_log(log_fp)
            results[str(secret_triple)]["original"] = og_rank
            open(log_fp, "w").close()

            #Check if each triple in a path is also in new data + remove adn test if so
            best_change = 0
            for triple in path_triples:
                if triple in new_data:
                    triple_count += 1
                    write_minus(train_fp, data_write, triple)

                    os.system("python my_rulen.py")# -d exp_v1_33/test -e exp_v1_33")
                    new_rank = parse_log(log_fp)
                    results[str(secret_triple)][str(triple)] = new_rank
                    if og_rank - new_rank > best_change:
                        best_change = og_rank - new_rank

                    open(log_fp, "w").close()
            best_changes.append(best_change)
        except:
            continue

    #print(count)
        
    with open(results_fp, "w") as fd:
        json.dump(results, fd)

    print(best_changes)
    print(np.mean(best_changes))


        
        
    

    
