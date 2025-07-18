from datasets import load_dataset
import random
import json
import copy
import string
import re
from collections import Counter

dataset_all = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
random.seed(1234)
dataset = list(dataset_all)

gen_path = "./"
dev_path = "partial_dev.json"
train_path = "partial_train.json"

with open(gen_path + dev_path, "r") as fp:
    dev = json.load(fp)

dev = dataset

with open(gen_path + train_path, "r") as fp:
    train = json.load(fp)
    
counts = set()
for example in dev:
  gold_index = -1
  for i, cont in enumerate(example["context"]["title"]):
    if cont == example["supporting_facts"]["title"][0]:
      gold_index = i
      break
  counts.update([len(example["context"]["sentences"][gold_index])])
  
  
def convert_example(ex):
  new_ex = copy.deepcopy(ex)
  new_ex["_id"] = new_ex["id"]
  del new_ex["id"]
  new_context = []
  for i, con in enumerate(example["context"]["title"]):
    
    if con == example["supporting_facts"]["title"][0]:
      gold_index = i
      gold_para = " ".join(example["context"]["sentences"][i])
      break
  for i in range(len(ex["context"]["title"])):
    new_context.append([ex["context"]["title"][i], ex["context"]["sentences"][i]])

  new_ex["context"] = new_context
  return new_ex
  
dumper = []
for example in dev:
  temp = convert_example(example)
  dumper.append(temp)

with open(gen_path + "new_dev.json", "w") as fp:
  json.dump(dumper, fp)
  
  
  
for max_len in counts:
  dumper = []
  out_file = gen_path + "Splits/" + str(max_len) + "dev.json"
  for example in dev:
    gold_index = -1
    for i, cont in enumerate(example["context"]["title"]):

      if cont == example["supporting_facts"]["title"][0]:
        gold_index = i
    if len(example["context"]["sentences"][gold_index]) < max_len:
      continue

    temp_ex = copy.deepcopy(example)
    temp_ex["context"]["sentences"][gold_index][max_len-1] = ""
    temp = convert_example(temp_ex)
    dumper.append(temp)
    
  print("{}: {}".format(max_len, len(dumper)))
  
  with open(out_file, "w") as fp: