#!/bin/bash
#SBATCH --cpus-per-task=4


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --qos=medium
#SBATCH --time=48:00:00
#SBATCH --output=./output/baseline.%a.out


eval "$(../../../NBFNet/miniconda3/bin/conda shell.bash hook)"
conda activate text_llama

export HF_HOME=/fs/nexus-scratch/nrolling/kg_experiments/


CUDA_VISIBLE_DEVICES=1 \
python train_beam_retriever.py \
--do_train \
--gradient_checkpointing \
--prefix \
retr_hotpot_beam_size2_large \
--model_name \
microsoft/deberta-v3-large \
--tokenizer_path \
microsoft/deberta-v3-large \
--dataset_type \
hotpot \
--train_file \
../hotpot_train_v1.1.json \
--predict_file \
../hotpot_dev_distractor_v1.json \
--train_batch_size \
8 \
--learning_rate \
2e-5 \
--fp16 \
--beam_size \
2 \
--predict_batch_size \
1 \
--warmup-ratio \
0.1 \
--num_train_epochs \
20 \
--mean_passage_len \
250 \
--log_period_ratio \
0.01 \
--accumulate_gradients \
8 \
--eval_period_ratio \
0.01




conda deactivate
