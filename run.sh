#!/bin/bash

module load pytorch/2.3.1
. /global/common/software/m4641/venv-2.3.1/bin/activate



export CUDA_DEVICE_MAX_CONNECTIONS=1
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))

export WORLD_SIZE=$GPUS
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
# these are specific to perlmutter's slingshot-11 network
#
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1

export MPICH_GPU_SUPPORT_ENABLED=0

#MODEL="tiiuae/falcon-7b"
#MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL="google/gemma-7b"
STRAT="axonn" #fsdp/axonn

SCRIPT="python -u -m litgpt finetune_full $MODEL  --data Alpaca --devices 4 --train.global_batch_size 32 --train.micro_batch_size 2 --num_nodes $NNODES --strategy $STRAT"

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 --ntasks-per-node=4 ./get_rank_from_slurm.sh $SCRIPT" 

echo $run_cmd
eval $run_cmd
set +x
