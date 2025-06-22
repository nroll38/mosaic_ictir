#!/bin/bash
#SBATCH --cpus-per-task=4


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --qos=medium
#SBATCH --time=48:00:00
#SBATCH --output=./output/eval.%a.out


eval "$(../../../NBFNet/miniconda3/bin/conda shell.bash hook)"
conda activate text_llama

export HF_HOME=/fs/nexus-scratch/nrolling/kg_experiments/


CUDA_VISIBLE_DEVICES=0 python test_model_tmp.py --type hotpot --reader_checkpoint ./reader_checkpoint.pt --retriever_checkpoint ./retriever_checkpoint.pt --file_path ./test.json --name "base"




conda deactivate
