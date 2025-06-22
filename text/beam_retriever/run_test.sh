#!/bin/bash
#SBATCH --cpus-per-task=4


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --qos=medium
#SBATCH --time=48:00:00
#SBATCH --output=./output/short.%a.out


eval "$(../../../NBFNet/miniconda3/bin/conda shell.bash hook)"
conda activate text_llama

export HF_HOME=/fs/nexus-scratch/nrolling/kg_experiments/


CUDA_VISIBLE_DEVICES=1 python train_beam_retriever.py --do_predict --model_name microsoft/deberta-v3-large --tokenizer_path microsoft/deberta-v3-large --dataset_type hotpot --train_file ../new_train.json  --predict_file ../new_dev.json --fp16 --beam_size 2 --predict_batch_size 1 --warmup-ratio 0.1 --mean_passage_len 250 




conda deactivate
