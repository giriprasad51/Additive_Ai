#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=DiasC
#SBATCH --partition=allgpu
#SBATCH --nodelist=node[8]
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=0-10:00:00
#SBATCH --error=./outputs/tem.%J.err
#SBATCH --output=./outputs/tem.%J.out



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

# srun torchrun --nnodes 1 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500     vgg19_torch_shard.py densenet121 MNIST   #dias_small.py # vgg19_modeldistest.py  #   #dis_layers_test1.py # 
# srun python -m viztracer --log_torch -o ./profiler_output/tracer_v100_nonpipelining1.json -m \
#     torch.distributed.run --nnodes 1 --nproc_per_node 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 vgg19_torch_shard.py 

# srun torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 normal_layers_test.py

# pytest normal_layers_test.py
# python normal_layers_test.py

python  vgg19_modeltest.py
# python  vgg19_modeltest_MHL.py
# /scratch/pusunuru/AI_repos/RNOCSLT/vgg19_modeltest_MHLpy
# python lenet5_modeltest_MHL.py