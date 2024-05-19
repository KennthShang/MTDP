#!/bin/bash
#SBATCH -o distill_log.out
#SBATCH -J distill
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=120:00:00
#SBATCH --export=all
#SBATCH -p team3
#SBATCH --gres=gpu:4



# if you want to run the code without deepspeed
python -u distillation.py --teacher1 esm.feat --teacher2 prottrans.feat --inputFA protein.fa --db MTDP_tokenizer/ --outpth out/

# if you want to run the code with deepspeed
#export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; deepspeed --num_gpus=4 distillation.py --deepspeed ds_config.json 

