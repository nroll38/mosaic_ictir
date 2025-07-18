from getpass import getpass
import os
import replicate
from openai import OpenAI
import time

import sys
import json
import re
import string
from collections import Counter
import pickle
from copy import deepcopy





os.environ["REPLICATE_API_TOKEN"] = "[REPLICATE KEY HERE]"
api_key = "[OPEN AI KEY HERE]"
os.environ["OPENAI_API_KEY"] ="[OPEN AI KEY HERE]"










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





import torch
from datasets import load_dataset

import replicate
import time
import random
import copy
import anthropic




def load_llama_model(model_path, tokenizer_path, max_length=2048):

    tokenizer = None
    model = None
    return tokenizer, model


def answer_question_claude(model, tokenizer, question, context, gp="", max_length=2048, preprompt_in=None):

    if not preprompt_in:
      preprompt = "Answer the following question using only the information in the provided target paragraph and context. Provide your reasoning first and then list factoid answer by itself on a new line. Finally, provide the most significant sentence from the \"Target Paragraph\" that explains your answer, prepended by a \"1.\".\n\n"
    else:
      preprompt = preprompt_in


    input_text = preprompt + f"Question: {question}\n\nTarget Paragraph: {gp}\n\nContext: {context}\n\nReasoning: "

    client = anthropic.Anthropic(
    api_key=claude_api_key,
    )

    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": input_text}
        ]
    )


    answer = message.content[0].text
    return answer


def answer_question(model, tokenizer, question, context, gp="", max_length=2048, preprompt_in=None):

    use_gpt = True
    
    if not preprompt_in:

      preprompt = "Answer the following question using only the information in the provided target paragraph and context. Provide your reasoning first and then list factoid answer by itself on a new line. Finally, provide the most significant sentence from the \"Target Paragraph\" that explains your answer, prepended by a \"1.\".\n\n"

    else:
      preprompt = preprompt_in


    input_text = preprompt + f"Question: {question}\n\nTarget Paragraph: {gp}\n\nContext: {context}\n\nReasoning: "

    if use_gpt:
        input = {
          "top_p": 0.9,
          "prompt": input_text,
          "min_tokens": 1,
          "temperature": 0.0,
          "max_tokens": 2000,
          "length_penalty": 3,
        }

        msg = [{"role": "user", "content": input_text}]
        client = OpenAI(api_key=api_key)
        answer = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=msg,
            temperature=0.0,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        answer=answer.choices[0].message.content
    else:
    for event in replicate.stream(
          "meta/meta-llama-3-8b",
          input=input
        ):

        print(event, end="")

        print("RUNNING")
        answer = replicate.run(
            "meta/meta-llama-3-70b-instruct",
            input=input,
            wait=True
        )
        answer = "".join

    return answer


def get_gpt_answer(temp_split):

  ans_next = False
  temp_answer=""
  temp_sentences = []
  for ln, line in enumerate(temp_split):
    if temp_answer=="" and line.lower().endswith("answer:"):
      ans_next=True
    elif ans_next and len(line) > 0:
      temp_answer = temp_split[ln]
      ans_next = False
    elif "1." in line or "2." in line:
      temp_sentences.append(line[2:].strip())
      if (temp_sentences[-1].endswith("\"") and temp_sentences[-1][0]==("\"")) or (temp_sentences[-1].endswith("\'") and temp_sentences[-1][0]==("\'")):
        #print(sentences[-1][1:-1])
        temp_sentences[-1] = temp_sentences[-1][1:-1]


  return temp_answer, temp_sentences

def get_claude_answer(temp_split):

  ans_next = False
  temp_answer=""
  temp_sentences = []
  for ln, line in enumerate(temp_split):
    if temp_answer=="" and line.lower().endswith("answer:"):
      ans_next=True
    elif ans_next and len(line) > 0:
      temp_answer = temp_split[ln]
      ans_next = False
    elif "1." in line[:5] or "2." in line[:5]:
      if "1." in line[:5] and ln > 1:
        temp_answer = temp_split[ln-2]
      temp_sentences.append(line[2:].strip())
      if (temp_sentences[-1].endswith("\"") and temp_sentences[-1][0]==("\"")) or (temp_sentences[-1].endswith("\'") and temp_sentences[-1][0]==("\'")):
        #print(sentences[-1][1:-1])
        temp_sentences[-1] = temp_sentences[-1][1:-1]


  return temp_answer, temp_sentences



    return answer

# Main function to run question answering on a subset of HotpotQA
def run_qa_on_hotpotqa(model_path, tokenizer_path, out_path, redact=False):
    max_length = 8192
    model = None
    tokenizer = None
    total_delta_em = 0
    total_delta_f1 = 0

    total_nonzero_em = 0
    total_nonzero_f1 = 0

    in_gold = 0
    sent_count = 0
    overlap_count = 0
    max_em_drop = 0

    gen_path = "./"
    dataset_all = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    random.seed(1234)
    dataset = list(dataset_all)[:4000]

    print(len(dataset))
    answers = {}

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    temp_metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}


    start = time.time()
    bad_count = 0
    for count, example in enumerate(dataset):
      try:
        gold_index = 0
        for i, cont in enumerate(example["context"]["title"]):
          if cont == example["supporting_facts"]["title"][0]:
            gold_index = i
            gold_para = " ".join(example["context"]["sentences"][i])

            break

        question = example['question']
        contexts = []
        for i in range(len(example['context']["sentences"])):
          if i != gold_index:
            contexts.append(" ".join(example['context']["sentences"][i]))
        context = "\n\n".join(contexts)

        answer=""
        sentences=[]
        ans_next = False
        response = answer_question(model, tokenizer, question, context, gp=gold_para, max_length=max_length)
        split = "".join(response).split("\n")



        for ln, line in enumerate(split):
          if answer=="" and line.lower().endswith("answer:"):
            ans_next=True
          elif ans_next and len(line) > 0:
            answer = split[ln]
            ans_next = False
          if "1." in line or "2." in line:
            sentences.append(line[2:].strip())
            if (sentences[-1].endswith("\"") and sentences[-1][0]==("\"")) or (sentences[-1].endswith("\'") and sentences[-1][0]==("\'")):
              sentences[-1] = sentences[-1][1:-1]

        em, prec, recall, f1 = update_answer(metrics, answer, example["answer"])
        answers[example["id"]] = {}
        answers[example["id"]]["answer"] = example["answer"]
        answers[example["id"]]["pred_answer"] = answer
        answers[example["id"]]["r_answer"] = []

        answers[example["id"]]["em"] = em
        answers[example["id"]]["f1"] = f1

        answers[example["id"]]["r_sentences"] = ""
        answers[example["id"]]["r_em"] = ""
        answers[example["id"]]["r_f1"] = ""
        answers[example["id"]]["max_em_drop"] = ""
        answers[example["id"]]["max_id"] = ""
        answers[example["id"]]["sentences"] = ""
        answers[example["id"]]["delta_em"] = ""
        answers[example["id"]]["delta_f1"] = ""
        answers[example["id"]]["metrics"] = ""
        answers[example["id"]]["count"] = ""

        answers[example["id"]]["blockers"] = []
        answers[example["id"]]["selected"] = []

        delta_em = 0
        delta_f1 = 0

        golds_idd = []
        for sent_num, sent in enumerate(sentences):
          if sent in example["context"]["sentences"][gold_index]:
            for i, gs in enumerate(example["context"]["sentences"][gold_index]):
              if sent in gs:
                sent_id = i
                golds_idd.append(sent_id)
                in_gold += 1

        if em:
            redacted = False
            delta_em = 0
            delta_f1 = 0
            tmax_em_drop = 0
            gold_em = 0
            gold_f1 = 0
            gold_ems = []
            gold_f1s = []
            max_id = []
            supporting_index = []

            temp_contexts = deepcopy(example['context']["sentences"])
            for i in range(len(example["context"]["sentences"][gold_index])):
              temp_contexts[gold_index] = deepcopy(example["context"]["sentences"][gold_index])
              temp_contexts[gold_index][i] = "[REDACTED]"
              new_contexts = []
              for j in range(len(temp_contexts)):
                if j == gold_index:
                  temp_sent = " ".join(temp_contexts[j])
                  temp_sent = ' '.join(temp_sent.split())
                  gold_para = temp_sent
                  continue
                new_contexts.append(" ".join(temp_contexts[j]))
              temp_context = "\n\n".join(new_contexts)

              temp_response = answer_question(model, tokenizer, question, temp_context, gp=temp_sent, max_length=max_length)
              temp_split = "".join(temp_response).split("\n")

              ans_next = False
              temp_answer=""
              temp_sentences = []
              for ln, line in enumerate(temp_split):
                if temp_answer=="" and line.lower().endswith("answer:"):
                  ans_next=True
                elif ans_next and len(line) > 0:
                  temp_answer = temp_split[ln]
                  ans_next = False
                elif "1." in line or "2." in line:
                  temp_sentences.append(line[2:].strip())
                  if (temp_sentences[-1].endswith("\"") and temp_sentences[-1][0]==("\"")) or (temp_sentences[-1].endswith("\'") and temp_sentences[-1][0]==("\'")):
                    temp_sentences[-1] = temp_sentences[-1][1:-1]

              r_em, r_prec, r_recall, r_f1 = update_answer(temp_metrics, temp_answer, example["answer"])

              if em and not r_em:
                tmax_em_drop = True
                max_id.append(i)

                all_remove = False
                if (golds_idd and i in golds_idd) or all_remove == True:
                  gold_em = r_em
                  gold_f1 = r_f1
                  gold_ems.append(r_em)
                  gold_f1s.append(r_f1)
                  answers[example["id"]]["r_answer"].append(temp_answer)
                  answers[example["id"]]["selected"].append(i)

            delta_em = em - gold_em
            delta_f1 = f1 - gold_f1
            max_em_drop += tmax_em_drop
            for id in golds_idd:
              if id in max_id:
                overlap_count += 1
                break

            answers[example["id"]]["r_em"] = gold_ems
            answers[example["id"]]["r_f1"] = gold_f1s
            answers[example["id"]]["max_em_drop"] = max_em_drop
            answers[example["id"]]["max_id"] = max_id




            answers[example["id"]]["sentences"] = sentences
            answers[example["id"]]["delta_em"] = delta_em
            answers[example["id"]]["delta_f1"] = delta_f1


            total_delta_em += delta_em
            total_delta_f1 += delta_f1

            if delta_em != 0:
              total_nonzero_em += 1
            if delta_f1 != 0:
              total_nonzero_f1 += 1

        answers[example["id"]]["metrics"] = metrics
        answers[example["id"]]["count"] = count
        answers[example["id"]]["g_idd"] = copy.deepcopy(golds_idd)


        count+=1
        if count % 50 == 0:
            print("At {}".format(count))
            print("Running Time: {}".format(time.time()-start))
            print("Sent Count: {}".format(sent_count))
            with open(gen_path + out_path, "w") as fp:
              json.dump(answers, fp)

            print("Overlap Count: {}\t\tmax_drop: {}".format(overlap_count, max_em_drop))
            for k,v in metrics.items():
              print("{} : {}".format(k, v/(count)))
            print("Total delta EM: {}\t f1: {}\nCount delta em: {}\t Count delta f1: {}\nIn Gold: {}".format(total_delta_em, total_delta_f1, total_nonzero_em, total_nonzero_f1, in_gold))
            print("BC: {}".format(bad_count))
            print("-"*50)



      except Exception as e:
        bad_count += 1
        print(e)


run_qa_on_hotpotqa(model_path, tokenizer_path, out_path="results.json", redact = True)
