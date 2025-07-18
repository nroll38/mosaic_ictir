from datasets import load_dataset
import random
import json
import copy
import string
import re
from collections import Counter
import aiohttp

gen_path = "./"
dev_path = ""
gpt_one_gp = ""
gpt_two_gp = ""
gpt_two_nogp = ""



llama_one_gp = ""
llama_two_gp = ""
llama_two_nogp = ""



results = {"one_gp": {"path": gpt_one_gp},
         "two_gp": {"path": gpt_two_gp},
         "two_nogp": {"path": gpt_two_nogp},
           "llama_one_gp": {"path": llama_one_gp},
           "llama_two_gp": {"path": llama_two_gp},
           "llama_two_nogp": {"path": llama_two_nogp},

           }
           
           


dataset_all = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True,
                           storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
random.seed(1234)
dataset = list(dataset_all)
data_by_id = {}
for example in dataset:
  data_by_id[example["id"]] = example
  print(example)
  # break
  if "cosmere" in example["question"].lower() or "cosmere" in example["context"]["sentences"]:
    print(example)


for name, res in results.items():
  with open(gen_path + res["path"], "r") as fp:
    results[name]["data"] = json.load(fp)


shared = list(set(results["one_gp"]["data"].keys()) & set(results["two_gp"]["data"].keys()) & set(results["two_nogp"]["data"].keys()) & set(results["llama_one_gp"]["data"].keys()) & set(results["llama_two_gp"]["data"].keys()) & set(results["llama_two_nogp"]["data"].keys()))

shared = shared[:4000]
shared = set(shared)


counts = set()
for example in dataset:
  gold_index = -1
  for i, cont in enumerate(example["context"]["title"]):
    #print(cont)
    if cont == example["supporting_facts"]["title"][0]:
      gold_index = i
      #gold_para = " ".join(example["context"]["sentences"][i])
      break
  counts.update([len(example["context"]["sentences"][gold_index])])
  
  
 def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)


    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall, f1

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall, f1
    
 
 metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}

# Directory for beam results - one file for the number of each sentence removed (ex second sentence removed, 2_pred.json, etc.
temp_path = ""
beam_data = []
for i in range(1, 32):
    filename = f"{i}_pred.json"  
    try:
        with open(temp_path + filename, 'r') as file:
                content = json.load(file)
                beam_data.append(content)
        print(len(beam_data[-1]))
    except:
        beam_data.append({})
        
#unredacted prediction path
with open(temp_path + "base_pred.json", 'r') as fp:
  beam_base = json.load(fp)


kd = {}
for entry in dataset:
  kd[entry["id"]] = entry
  
  
em_count = 0
red_count = 0
beam_ems = {}
metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}

overlap = 0
fr = 0
pr = 0

for id in beam_base.keys():
  if id not in shared:
    continue
  actual = kd[id]["answer"]
  em, prec, recall, f1 = update_answer(metrics, beam_base[id], actual)

  if em == False:
    continue
  em_count += 1
  beam_ems[id] = []

  mt = False
  for i in range(len(beam_data)):
    if id not in beam_data[i]:
      continue
    em, prec, recall, f1 = update_answer(metrics, beam_data[i][id], actual)

    
    if em == False:

      mt = True
      if len(beam_ems[id])==0:
        red_count += 1
        pr += 1

      beam_ems[id].append(i)


  if mt == False:
    fr += 1


print(em_count)
print(red_count)



metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}





#RNN path
temp_path = ""
rnn_data = []
for i in range(1, 32):
    filename = f"{i}_pred.json"
    try:
        with open(temp_path + filename, 'r') as file:
                content = json.load(file)
                rnn_data.append(content)
    except:
        rnn_data.append({})
with open(temp_path + "all_pred.json", 'r') as file:
  rnn_base = json.load(file)

print(rnn_base.keys())




em_count = 0
red_count = 0
rnn_ems = {}

overlap = 0
pr = 0
fr = 0

for id in rnn_base["answer"].keys():
  #print(base['answer'][id])
  #print(id)
  #print(beam_base[id])
  #best[id] = -1
  if id not in shared:
    continue

  actual = kd[id]["answer"]
  
  em, prec, recall, f1 = update_answer(metrics, rnn_base["answer"][id], actual)
  
  if em == False:
    continue
  em_count += 1
  rnn_ems[id] = []

  mt = False
  for i in range(len(rnn_data)):
    if id not in rnn_data[i]["answer"]:
      continue
    em, prec, recall, f1 = update_answer(metrics, rnn_data[i]["answer"][id], actual)
   
    if em == False:
      mt = True
      if len(rnn_ems[id])==0:
        red_count += 1

        pr += 1
        for k,v in data_by_id[id].items():
          if k == "context":
            for j, t in enumerate(v["title"]):
              if t == data_by_id[id]["supporting_facts"]["title"][0]:
                for sentence in v["sentences"][j]:
                  print(sentence)
          else:
            print("{}:{}".format(k, v))

          print("*"*20)
      rnn_ems[id].append(i)
  if mt == False:
    fr += 1

      

print(em_count)
print(red_count)




count_good = 0
combined_results = {}
overall_data = {}
match_12 = 0
match_23 = 0
match_13 = 0
match_123 = 0


count_pr = [0, 0, 0]
count_fr = [0, 0, 0]
count_overlap = [0, 0, 0]
total_delta = 0
count_delta = [0, 0, 0]
count_em = 0
failed_rem = 0


f_counts = []
all_em_count = []
possible = []

per_question = {}
per_model = []
per_model_em = []
per_model_redact = []
per_model_idx = {}

per_model_select = []


counts = [0, 0, 0, 0, 0, 0]



if True:


    for name in results.keys():
      per_model_idx[name] = len(per_model)
      per_model.append({})
      per_model_em.append({})
      per_model_select.append({})
      per_model_redact.append({})
      possible.append(0)
      f_counts.append(0)
      all_em_count.append(0)

    per_model_idx["beam"] = len(per_model)
    per_model_em.append({})
    per_model_select.append({})

    print(results.keys())
    name = list(results.keys())[0]

    for id in results[name]["data"].keys():
      if id not in shared:
        continue
      q_results = []
      combined_results[id] = {}




      for local_name in results.keys():
        try:
          q_results.append(results[local_name]["data"][id])
        except:
          q_results.append(None)


      ems = []
      r_ems = []
      red_sentences = []
      max_ids = []
      flipped = []
      target = None
      f = False

      for i in range(len(q_results)):

        if q_results[i] == None:
          continue
        ems.append(q_results[i]["em"])
        if q_results[i]["em"] == True:
          all_em_count[i] += 1
          per_model_em[i][id] = q_results[i]["max_id"]
        r_ems.append(q_results[i]["r_em"])
        red_sentences.append(q_results[i]["selected"])

        if q_results[i]["sentences"]:
          target =  q_results[i]["sentences"][0]
          for para in kd[id]["context"]["sentences"]:
            for cnt, sentence in enumerate(para):
              if target in sentence:
                if cnt in q_results[i]["max_id"]:
                    counts[i]+=1
                break

        
        per_model_select[i][id] = q_results[i]["selected"]

        max_ids.append(q_results[i]["max_id"])
        
        if q_results[i]["max_id"]:
          possible[i] += 1

        flipped.append(f)
        
        
        
import csv


def fix_vals(m):
  if type(m) is list:
    return m
  else:
    return [m]

def mod_count_matches(m1, m2, m3={}):
  count_em = 0
  count_red = 0

  for id in m1.keys():
    if id in m2 or id in m3:
      count_em += 1
      usid = None
      if id in m2 and id in m3:
        vals2 = fix_vals(m2[id])
        vals3 = fix_vals(m3[id])
        for sid in vals2:
          if sid in vals3:
           usid = sid
           break
        if not usid:
          if vals2:
            usid = vals2[0]
          elif vals3:
            usid = vals3[0]
      elif id in m2 and m2[id]:
        vals2 = fix_vals(m2[id])
        usid = vals2[0]
      elif id in m3 and m3[id]:
        vals3 = fix_vals(m3[id])
        usid = vals3[0]
      else:
        continue

      vals1 = fix_vals(m1[id])
      if vals1 and usid in vals1:

        count_red += 1


  return count_em, count_red

with open('rand_em_match.csv', 'w', newline='') as csvfile_em:
    em_writer = csv.writer(csvfile_em, delimiter=',')
    random.seed(1234)
    with open("red_match_models.csv", 'w', newline='') as csvfile_mod:
        model_writer = csv.writer(csvfile_mod, delimiter=',')
        with open('rand_red_match.csv', 'w', newline='') as csvfile_red:
            red_writer = csv.writer(csvfile_red, delimiter=',')
            output = ["Model", "MAX"]
            output += list(per_model_idx.keys())
            em_writer.writerow(output)
            red_writer.writerow(output)
            model_writer.writerow(output)
            print(len(output))
            for model in per_model_idx.keys():
              if "none" in model:
                continue

              output_em = [model]
              output_red = [model]


              output_em_model = ["",""]
              output_red_model = [model,""]

              count_em, count_red = mod_count_matches(per_model_em[per_model_idx[model]], per_model_em[per_model_idx[model]])
              output_em.append(count_em)
              output_red.append(count_red)


              for model2 in per_model_idx.keys():
                if "none" in model2:
                  continue
                count_em = 0
                count_red = 0


                for model3 in per_model_idx.keys():
                  if model3 == model or model3 == model2 or "none" in model3:
                    continue

                  t_count_em, t_count_red = mod_count_matches(per_model_em[per_model_idx[model]], per_model_em[per_model_idx[model2]], per_model_em[per_model_idx[model3]])
                  if t_count_em > count_em:
                    count_em = t_count_em
                    best_model_em = model3
                  if t_count_red > count_red:
                    count_red = t_count_red
                    best_model_red = model3

                output_em.append(count_em)
                output_red.append(count_red)


                output_red_model.append(best_model_red)


              em_writer.writerow(output_em)
              red_writer.writerow(output_red)

              model_writer.writerow(output_red_model)
      