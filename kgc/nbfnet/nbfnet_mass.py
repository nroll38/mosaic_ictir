import numpy as np
from tqdm import tqdm
import csv
import random
import copy
import os
import json
import sys
import argparse
import time

from script import run

import torch
import pprint
from torchdrug import core, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util


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
        for i, row in enumerate(data_write):
            if row != triple:
                writ.writerow(row)
            else:
                if i < len(data_write)-1:
                    writ.writerow(data_write[i+1])
                else:
                    writ.writerow(data_write[0])

#grabs RR from GraIL log
def parse_log(fp):
    with open(fp, "r") as fd:
        for line in fd.readlines():
            #print(line)
            continue
        
        
    print("FINAL LINE SPLIT: {}".format(line.split(":")))
    return float(line.split(":")[3])


#modified from nbfnet/utils
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", default="config/inductive/fb15k237_oracle.yaml")
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    #parser.add_argument("-g", "--gpus", required=False, default="[0]")
    #parser.add_argument("-v", "--version", required=False, default="v1")

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = util.detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars
                
if __name__ == "__main__":
    
    dp = os.path.expanduser('~') + "/datasets/knowledge_graphs/grail_sets/"
    directory = dp
    data_fp = dp+"test/train_og.txt"
    secret_fp = dp+"secrets.txt"
    new_data_fp = dp+"new.txt"

    train_fp = dp+"test/train.txt"
    test_fp = dp+"test/test.txt"


    
    train_og_fp = directory + "train/train_og.txt"
    valid_og_fp = directory + "train/valid_og.txt"
    test_og_fp = directory + "test/test_og.txt"
    # secret_fp = directory + "secrets.txt"
    # new_data_fp = directory + "/new.txt"

    train_fp = directory + "train/train.txt"
    valid_fp = directory + "train/valid.txt"
    test_fp = directory + "test/test.txt"
    

    log_fp = "./log_rank_test.txt"

    results_fp = "./oracle_results.json"
    changes_fp = "./oracle_changes.json"

    #model_dp = "
    
    #new_secrets = "./data/exp_v1_33/secrets.txt"
    results = {}
    best_changes = []


    # #from nbfnet run.py
    # # args, vars = parse_args()
    # # cfg = util.load_config(args.config, context=vars)
    # # working_dir = util.create_working_directory(cfg)

    # # torch.manual_seed(args.seed + comm.get_rank())

    # # logger = util.get_root_logger()
    # # if comm.get_rank() == 0:
    # #     logger.warning("Config file: %s" % args.config)
    # #     logger.warning(pprint.pformat(cfg))

    # # dataset = core.Configurable.load_config_dict(cfg.dataset)
    # # solver = util.build_solver(cfg, dataset)

    # # run.train_and_validate(cfg, solver)
    # # run.test(cfg, solver)





                        
    # data, data_s, data_write = load_data(data_fp)
    # secrets = load_set(secret_fp)
    # new_data = load_set(new_data_fp)

    # all_train_data = load_set(data_fp)

    # #improve secrets:
    # # try:
    # #     with open(results_fp, "r") as fd:
    # #         temp = json.load(fd)
    # #     with open(changes_fp, "r") as fd:
    # #         temp_2 = json.load(fd)
    # #     if temp:
    # #         results = temp
    # #         best_changes = temp_2
    # # except:
    # #     pass


    rel_types = set()
    with open("rel_types.txt", "r") as fp:
        lines = fp.readlines()
        for line in lines:
            rel_types.update([line.strip()])

    print(len(rel_types))

    #for (s,r,o) in secrets:
    #    secret_triple = (s,r,o)

    rel_types = list(rel_types)
    rel_types.sort()


    #print("BAD RELS: \n{}\n{}\n{}\n{}\n".format(rel_types[8], rel_types[9], rel_types[11], rel_types[15]))

    #exit()
    my_log_fp = "mass_log.txt"

    
    
    # for (s,r,o) in secrets:
    #     secret_triple = (s,r,o)
    #     if str(secret_triple) in results:
    #         continue

    with open(my_log_fp, "w") as fp:
        print(my_log_fp)
        
    for k, r in enumerate(rel_types):
        paths = {}
        prev_path = ()
        triple_count = 0
        print("On {}/{}: {}".format(k, len(rel_types), r))


        with open(valid_og_fp, "r") as fp_1:
            lines = fp_1.readlines()

        with open(valid_fp, "w") as fp_out:
            for line in lines:
                newline = line.split("\t")
                if newline[1] != r:
                    continue

                fp_out.write(line)


    
        #fix test
        with open(test_og_fp, "r") as fp_1:
            lines = fp_1.readlines()

        with open(test_fp, "w") as fp_out:
            for line in lines:
                newline = line.split("\t")
                if newline[1] != r:
                    continue
                fp_out.write(line)

        

        
        # results[str(secret_triple)] = {}
        # #ensure test, train  files initialized to full graph and target secret; clear log
        # write_minus(test_fp, [secret_triple], "")
        # write_minus(train_fp, data_write, "")
        # #open(log_fp, "w").close()


        # #get triples for each component of a path IOT ID if a valid new triple for potential break
        # paths = {}
        # prev_path = ()
        # triple_count = 0
        # find_surrounding_nodes_triples(data_s, s, o, 0, 3, paths, prev_path)
        # path_triples = get_triples_paths(paths, s, o)

        #os.system("python script/run.py -c config/inductive/fb15k237.yaml --gpus [0] --version v1 > "+log_fp)
        
        try:
            start_time = time.time()
            #os.system("python script/run.py -c config/inductive/fb15k237.yaml --gpus [0] --version v1 > "+log_fp)
            os.system("python script/run.py -c config/inductive/fb15k237_og_mass.yaml --gpus [0] --version v1 > "+log_fp)
            train_time = time.time() - start_time
            og_rank = parse_log(log_fp)
            print("OG time: {}".format(train_time))
            #exit()
            #open(log_fp, "w").close()

            #Check if each triple in a path is also in new data + remove adn test if so
            #os.system("python script/run.py -c config/inductive/fb15k237_oracle.yaml --gpus [0] --version v1 > "+log_fp)

            with open(test_fp+"_rels", "w") as fp_out:
                fp_out.write(r)
            
            start_time = time.time()
            os.system("python script/run.py -c config/inductive/fb15k237.yaml --gpus [0] --version v1  > "+log_fp)
            train_time = time.time() - start_time

            os.system("rm " + test_fp+"_rels")
            os.system("rm -r ~/experiments/InductiveKnowledgeGraphCompletion/FB15k237Inductive/NBFNet/2024-09-04*")
            mod_rank = parse_log(log_fp)

            with open(my_log_fp, "a") as fp:

                fp.write("{}\t{}\t{}\t{}\t{}\n".format(r, k, mod_rank, og_rank, train_time))
            

            continue

        
        except:
            #exit()
            continue

        
    

    
#0, 0, 0, 0, 0, 0, 0, 0, 0.07882006, 0, 0, 0, 0, 0, 0.03217300000000001, 0]
