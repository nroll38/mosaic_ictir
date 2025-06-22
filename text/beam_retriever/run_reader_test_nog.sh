#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa6000:1

#SBATCH --qos=high
#SBATCH --time=24:00:00
#SBATCH --output=./output/readertest.%a.out

#SBATCH --mem=64G    
#SBATCH --cpus-per-task=8

eval "$(../../../NBFNet/miniconda3/bin/conda shell.bash hook)"
conda activate text_llama

export HF_HOME=/fs/nexus-scratch/nrolling/kg_experiments/


CUDA_VISIBLE_DEVICES=0 python train_reader.py --do_predict --init_checkpoint ./reader_checkpoint.pt --prefix hotpot_reader_deberta_large --model_name microsoft/deberta-v3-large --tokenizer_path microsoft/deberta-v3-large --dataset_type hotpot --train_file ../hotpot_train_v1.1.json --predict_file ../data/new_dev_nog_all.json --train_batch_size 8 --learning_rate 1e-5 --fp16 --max_seq_len 1024 --num_train_epochs 16 --predict_batch_size 1 --warmup-ratio 0.1 --log_period_ratio 0.01 --eval_period_ratio 0.3
mv reader_pred.json preds/all_pred_nog.json



conda deactivate
