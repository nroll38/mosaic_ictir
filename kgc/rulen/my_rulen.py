import numpy as np
from tqdm import tqdm
import csv
import random
import copy
import argparse
import json
import statistics

import test_ranking

def load_data(fp):
    data = {}
    data_s = {}
    data_o = {}

    with open(fp, "r") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            (s, r, o)  = row
            
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


            if o not in data_o:
                data_o[o] = {}
            if r not in data_o[o]:
                data_o[o][r] = [o]
            else:
                data_o[o][r].append(s) 
    
    return data, data_s, data_o

def load_set(fp):

    ret_set = set()
    with open(fp, "r") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            (s, r, o)  = row
            ret_set.add((s,r,o))
    return ret_set

#randomly select target tuple
def select_target(data):

    r, s_dict = random.choice(list(data.items()))
    s, o_list = random.choice(list(s_dict.items()))
    o = random.choice(o_list)
    
    
    return (s,r,o)

#given s, target, steps, find surrounding nodes and recusively call until target found or max steps(k) reached
def find_surrounding_nodes(data_s, s, target_o, steps, k, paths, prev_path):
    if s not in data_s:
        return
    out = []
    for r, o_list in data_s[s].items():
        for o in o_list:
            #print("PREV PATH: {}".format(len(prev_path)))
            cur_path = prev_path + (r,)
            #print("CUR PATH: {}".format(len(cur_path)))
            #print("CUR PATH: {}".format(cur_path))
            if o == target_o:
                path = tuple(cur_path)
                #print("PATH: {}".format(path))
                if path not in paths:
                    paths[path] = 1
                else:
                    paths[path] += 1
                continue
            if steps < k-1:
                find_surrounding_nodes(data_s, o, target_o, steps + 1, k, paths, cur_path)
                

#given s, target, steps, find surrounding nodes and recusively call until target found or max steps(k) reached
def find_surrounding_nodes_triples(data_s, s, target_o, steps, k, paths, prev_path):
    if s not in data_s:
        return
    
    for r, o_list in data_s[s].items():
        for o in o_list:
            #print("PREV PATH: {}".format(len(prev_path)))
            cur_path = prev_path + (s,)
            #print("CUR PATH: {}".format(len(cur_path)))
            #print("CUR PATH: {}".format(cur_path))
            if o == target_o:
                path = tuple(cur_path)
                #print("PATH: {}".format(path))
                if path not in paths:
                    paths[path] = 1
                else:
                    paths[path] += 1
                continue
            if steps < k-1:
                find_surrounding_nodes_triples(data_s, o, target_o, steps + 1, k, paths, cur_path)
                
                    
    

#given r, random sample relations to find path prominence
def random_sample_relation(data, data_s, relation, sample_no=10, k = 2):
    rel_dict = data[relation]
    s_list = list(rel_dict.keys())
    
    if sample_no == 0:
        sample_no = len(s_list)
    
    #TODO: SO_LIST for multiple o's with same sr
    if len(s_list) == 0:
        return {}
    chosen_s = random.sample(s_list, min(sample_no, len(s_list)))

    
    paths = {}
    
    

    for s in chosen_s:
        cur_path = ()
        steps = 0
        target_o = rel_dict[s][0]
        
        find_surrounding_nodes(data_s, s, target_o, steps, k, paths, cur_path)
        #visited = set()
        #visited.add(s)
        #target_o = rel_dict[s][0]
        #depth = 0

        
        #for r in data_s[s].keys():
            
            
            
        
    

        
    #print(len(paths))
    pks = paths.keys()
    for k in pks:
        paths[k] = min(paths[k]/len(chosen_s), 1.0)
    return paths

def path_check(s, path, hop, data_s):
    found = []
    if len(path) == hop:
        return [s]
    r = path[hop]
    if s in data_s and r in data_s[s]:
        for o in data_s[s][r]:
            found += path_check(o, path, hop + 1, data_s)
    return found

def rule_check(s, o, path, hop, data_s):
    found = False

    #print("Hop {} of {}".format(hop, len(path)))
    #print("S: {}\t O: {}".format(s, o))
    if len(path) == hop:
        if s == o:
            #print("TRUE")
            return True
        else:
            #print("False")
            return False

        #return [s]
    r = path[hop]
    if s in data_s and r in data_s[s]:
        for new_o in data_s[s][r]:
            #print("Checking {} for {}".format(new_o, r))
            found = found or rule_check(new_o, o, path, hop + 1, data_s)
    return found


#given s and o, should r exist?
def rulen_mod(s, r, o, paths, data, data_s,  neg_samples_dict, max_sub=0):
    all_found = []
    rules_found = {}
    match_secret = []
    ranks = []
    secret = (s, r, o)


    #head/tail negative samples
    for style, neg_samples in neg_samples_dict[secret].items():
        #print(neg_samples)

        rules_found = {}
        match_secret = []
        count = 0
        for path in paths:
            #print(path)
            for sample in neg_samples:
                #print(sample)
                if sample not in rules_found:
                    rules_found[sample] = []
                (new_s, new_r, new_o) = sample
                found = rule_check(new_s, new_o, path, 0, data_s)
                # if found:
                #     print(secret)
                #     print(sample)
                #     print("*"*20)
                count += 1
                #print("*"*20)
                #if count > 2:
                #    exit()
                rules_found[sample].append(found)
                #if found:
                #    all_found.append((found, path, paths[path]))
                #print(paths[path])

        #all_found.sort(key = lambda x: x[2], reverse=True)
        #print("PATHS: {}".format(len(paths)))
        #print("AF: {}".format(len(all_found)))
        #print(rules_found)

        # for k, v in rules_found.items():
        #     print (k)
        #     print(v)
        #     print("*"*20)
        #     break

        #need to break ties around the secret in the rankings, rest don't matter
        
        #TEMP TESTING
        # secret = "secret"
        # rules_found = {"secret": [True, False, True, False],
        #                "a": [True, False, True, False],
        #                "b": [False, False, True, False],
        #                "c": [True, True, False, False],}


        # check = {}
        # test_set = set()
        # for k, v in rules_found.items():
        #     #print(v)

        #     if len(v) not in check:
        #         check[len(v)] = 0
        #     check[len(v)] += 1
        # for i in neg_samples_dict[secret]["head"]:
        #     #print(i)
        #     if i in test_set:
        #         print("DUPLICATE: {}".format(i))
        #         print("SECRET: {}".format(secret))
        #     test_set.add(i)
        # print(check)
        #print(len(neg_samples_dict[secret]["head"]))

        
        
        sorting_rules_list = list(rules_found.items())
        match_secret = list(rules_found.items())
        
        above = []
        #print(match_secret)
        #print(len(match_secret[0][1]))
        for i in range(len(match_secret[0][1])):
            
            match_secret.sort(key=lambda x:  x[1][i], reverse = True)
            #print(match_secret)

            secret_found = rules_found[secret][i]
            #print(secret_found)

            temp_match = []
            for (k, v) in match_secret:
                if secret_found == v[i]:
                    temp_match.append((k, v))
                elif not secret_found and v[i] == True:
                    above.append((k,v))
            i+=1
            match_secret = temp_match
            if len(match_secret) < 2:
                break

        # print(match_secret)

        
        #rint(above)
        if len(match_secret) > 1:
            random.shuffle(match_secret)
            for i, entry in enumerate(match_secret):
                if entry[0] == secret:
                    break
                    
        ranks.append(1/(len(above)+ i + 1))
        #print(ranks)


    return np.mean(ranks)
            
    match_secret = []
    top_choice = set()
    second = []

    top = 0
    second_score = 0
    num_rem = 0
    j = 0
    
    found_list = []

    for found, path, num in all_found:
        #print(found)
        #print(path)
        #print(num)
        if o in found:
            found_list.append((found, path, num))
            if top == 0:
                top = num
            
            
    #print(top_choice)
    #print(top)
    #print(o)

    ret = False
    if o in top_choice:
        ret = True
    
    return top, found_list



def remove_r(data, data_s, s, path, o):

    rem_link = path[0]

    del data[rem_link][s]
    del data_s[s][rem_link]
    
    #print("DEL")
    return data, data_s

def find_rules(data, data_s, rs, k, sample_no=0):
    rules = {}
    for r_check in rs:
        #print("On rule {}".format(r_check))
        if r_check not in rules:
            paths = random_sample_relation(data, data_s, r_check, sample_no=sample_no, k = k)

            sorted_paths = {k: v for k, v in sorted(paths.items(), key=lambda item: item[1], reverse=True)}

            if (r_check,) in sorted_paths:
                del sorted_paths[r_check,]
            rules[r_check] = sorted_paths

        #print(sorted_paths)
    return rules

def get_negative_samples(data_fp, secrets_fp, rel2id_fp):
    fps= {}
    fps["graph"] = data_fp
    fps["links"] = secrets_fp

    neg_triple_out = {}

    with open(rel2id_fp, "r") as fd:
        rel2id = json.load(fd)
    
    
    adj_list, _, triples, _, _, id2entity, id2relation = test_ranking.process_files(fps, rel2id, False)
    neg_triples = test_ranking.get_neg_samples_replacing_head_tail(triples["links"], adj_list, num_samples=70)

    #print(rel2id)
    

    for entry in neg_triples:
        
        for k, v in entry.items():
            #print(k)
            checker = set()
            for i, triple in enumerate(v[0]):
                if len(checker) == 50:
                    break
                out = ""
                #print(triple)
                if tuple(triple) in checker:
                    continue
                checker.add(tuple(triple))
                    
                triple_text = (id2entity[triple[0]], id2relation[triple[2]], id2entity[triple[1]])
                
                if k == "head" and i == 0:
                    cur_secret = triple_text
                    neg_triple_out[cur_secret] = {}
                if k not in neg_triple_out[cur_secret]:
                    neg_triple_out[cur_secret][k] = []
                neg_triple_out[cur_secret][k].append(triple_text)
                    
                    

                
                
                #break
            #break
            #print(neg_triples[0])
    #print(neg_triple_out[cur_secret].keys())
    return neg_triple_out

def get_centrality(secrets, data_s, data_o, centrality_fp, k=3):

    center_stats = {}
    for secret in secrets:
        #print("SECRET: {}".format(secret))
        center_stats[str(secret)] = (0,0)
        paths = {}
        subgraph_stats = []
        cur_path = ()
        local_nodes = set()
        (s, r, o) = secret
        find_surrounding_nodes_triples(data_s, s, o, 0, k, paths, cur_path)

        for path in paths:
            #print(path)
            for i in range(len(path)):
                #print(path[i])
                local_nodes.add(path[i])
        
        for node in local_nodes:
            in_links = 0
            out_links = 0
            if node in data_o: 
                in_links = len(data_o[node])

            if node in data_s:
                out_links = len(data_s[node]) 
            subgraph_stats.append(in_links + out_links)
        if subgraph_stats:
            center_stats[str(secret)] = (max(subgraph_stats), statistics.median(subgraph_stats))
        #print(center_stats[str(secret)])


    with open(centrality_fp, "w") as fd:
        json.dump(center_stats, fd)
    
                
        
                
    


if __name__ == "__main__":
    #data_fp = "./data/FB15K237/train.txt"
    rules_data_fp = "./data/exp_v1_33_select/train.txt"#"./data/exp_v1_33_select/train/train.txt"
    data_fp = "./data/exp_v1_33_select/test/train.txt"
    #files = ["./data/FB15K237/train.txt", "./data/FB15K237/valid.txt", "./data/FB15K237/test.txt"]
    secrets_fp = "./data/exp_v1_33_select/test/test.txt"
    all_secrets_fp = "./data/exp_v1_33_select/secrets.txt"
    rules_dict = "./rulen_rules.json"
    rel2id_fp = "./data/exp_v1_33_select/relation2id.json"
    output_fp = "./rulen_mrr.txt"
    centrality_fp = "./experiments/exp_v1_33_select/centrality.json"



    # rules_data_fp = "./data/exp_v1_33/train/train.txt"
    # data_fp = "./data/exp_v1_33/test/train.txt"
    # #files = ["./data/FB15K237/train.txt", "./data/FB15K237/valid.txt", "./data/FB15K237/test.txt"]
    # secrets_fp = "./data/exp_v1_33/test/test.txt"
    # all_secrets_fp = "./data/exp_v1_33/secrets.txt"
    # rules_dict = "./rulen_rules.json"
    # rel2id_fp = "./data/exp_v1_33/relation2id.json"
    # output_fp = "./rulen_mrr.txt"
    # centrality_fp = "./experiments/exp_v1_33/centrality.json"
    
    data, data_s, data_o = load_data(data_fp)
    rules_data, rules_data_s, _ = load_data(rules_data_fp)
    secrets = load_set(secrets_fp)
    all_secrets = load_set(all_secrets_fp)
    random.seed(123)
    count = 0
    ten_count = 0
    scores = []
    ranks = []
    rem1 = 0
    rem2 = 0
    rules = {}

    rs = data.keys()

    sub = 0
    k = 3

    found_dict = {}
    output = []


    #TEMP
    get_centrality(all_secrets, data_s, data_o, centrality_fp)
    #exit()
    negative_samples = get_negative_samples(data_fp, secrets_fp, rel2id_fp)

    #Gen Rules
    try:
        with open(rules_dict, "r") as fd:
            rules = json.load(fd)
    except:
        rules = find_rules(data, data_s, rs, k)
    

    #EDIT
    #for i in tqdm(range(100)):
    for secret in secrets:
        ret = None
        cur_sub = sub

        top_rule_list = []
        (s,r,o) = secret


        output.append(rulen_mod(s, r, o, rules[r], data, data_s, negative_samples, max_sub=0))
    
    
    with open(output_fp, "w") as fd:
        for out in output:
            fd.write(str(out))


    #center = get_centrality(data_s, data_o)
        
    # if False:
    #     #print(rules.keys())
    #     while(ret==None):
            

            
    #         #data = copy.deepcopy(o_data)
    #         #data_s = copy.deepcopy(o_data_s)
    #         (s,r,o) = secret#select_target(data)

    #         #not in data
    #         # data[r][s].remove(o)
    #         # if len(data[r][s]) < 1:
    #         #     del data[r][s]

    #         # data_s[s][r].remove(o)
    #         # if len(data_s[s][r]) < 1:
    #         #     del data_s[s][r]

    #         all_found_list = {}
            
    #         #print("S: {}\t O: {}\t R: {}".format(s,o, r))
    #         #print(data_s['/m/0c8wxp'])

    #         rulen_mod(s, r, o, rules[r], data, data_s, negative_samples, max_sub=0)
            
    #         exit()
    #         for r_check in rs:
    #             #print("On rule {}".format(r_check))
    #             if r_check not in rules:
    #                 paths = random_sample_relation(data, data_s, r_check, sample_no=30, k = k)

    #                 sorted_paths = {k: v for k, v in sorted(paths.items(), key=lambda item: item[1], reverse=True)}

    #                 if (r_check,) in sorted_paths:
    #                     del sorted_paths[r_check,]
    #                 rules[r_check] = sorted_paths

    #             #print(sorted_paths)


    #             #res, num, rem = rulen_mod(s, r, o, rules[r], data, data_s, max_sub=0)
    #             num, found_list = rulen_mod(s, r, o, r_check, rules[r_check], data, data_s, negative_samples, max_sub=0)
    #             top_rule_list.append((r_check, num))
    #             if found_list:
    #                 #print("FOUND LIST: {}".format(found_list))
    #                 all_found_list[r_check] = found_list

    #         top_rule_list.sort(key=lambda x: x[1], reverse=True)
    #         for i, (r_rank, score) in enumerate(top_rule_list):
    #             if r == r_rank:
    #                 #print("R at rank {}".format(i))
    #                 break
    #         if cur_sub > 0 and i == 0:
    #             cur_sub -=1

    #             #print(all_found_list)
    #             try:
    #                 data, data_s = remove_r(data, data_s, s, all_found_list[r][0][1], o)
    #             except:
    #                 continue
                
    #         else:
    #             ret = True
    #             if i  == 0:
    #                 count += 1
    #             if i < 10:
    #                 ten_count += 1

    #     ranks.append(i)
    #     scores.append(score)
        

    # print("Correct: {}".format(count/100))
    # print("Top 10: {}".format(ten_count/100))
    # print("Total Rules: {}".format(len(rs)))
    # print("Avg Score: {}".format(np.mean(scores)))
    # print("Std dev: {}".format(np.std(scores)))
    # print("Avg Ranks: {}".format(np.median(ranks)))
    # print("Std dev: {}".format(np.std(ranks)))
    
    # #print("rem1: {}".format(rem1))
    # #print("rem2: {}".format(rem2))


