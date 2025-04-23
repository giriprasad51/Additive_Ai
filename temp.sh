#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=DiasC
#SBATCH --partition=allgpu
#SBATCH --nodelist=node[3,8]
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --error=./outputs/tem.%J.err
#SBATCH --output=./outputs/tem.%J.out
#SBATCH -A secvss_hip


# cd /home/mas/23/cdssona/projectfiles/diasDNN

scontrol show hostnames
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_arrays[0]}
echo "head node $head_node"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes_array[0]}" hostname --ip-address)

echo "Node IP: $head_node_ip"
export LOGLEVEL=INFO
# Add this to help with GPU assignment
CUDA_LAUNCH_BLOCKING=1

srun torchrun --nnodes 2 --nproc_per_node 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 dis_layers_test.py 

# srun torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 normal_layers_test.py

# pytest normal_layers_test.py
# python normal_layers_test.py

# python  vgg19_modeltest.py
# python lenet5_modeltest_MHL.py