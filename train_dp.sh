#!/bin/sh
#SBATCH --job-name=mirror_mask_and_norm
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000073
#SBATCH -G 2
#SBATCH --cpus-per-task=96
#SBATCH --mem=1024G
#SBATCH --error=outputs/mirror_mask_and_norm/train_dp.err
#SBATCH --output=outputs/mirror_mask_and_norm/train_dp.out
run_name="mirror_mask_and_norm"
if [ -z "$SLURM_GPUS" ]; then
    export SLURM_GPUS=$((SLURM_GPUS_PER_NODE*SLURM_NNODES))
fi

working_dir=/scratch/m000073/yihuai/robotics/repositories/policies/imitation-learning-policies/prior_works/universal_manipulation_interface



# root_dir=$(dirname $(realpath $0))
# cd $root_dir
cd $working_dir
python_path=$(conda info --base)/envs/umi/bin/python
accelerate_path=$(conda info --base)/envs/umi/bin/accelerate

shm_dir=/dev/shm/uva/umi_data

local_dataset_dir=/scratch/m000073/uva/umi_data


. $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
conda activate umi
which python

set -e

additional_arguments=""
additional_arguments="$additional_arguments task.dataset.dataloader_cfg.num_workers=48 task.dataset.dataloader_cfg.batch_size=192"
dataset_names=cup_arrangement_0,towel_folding_0,mouse_arrangement_0
lz4_data_path=/scratch/m000073/uva/umi_data/lz4
$python_path scripts/extract_umi_data.py $dataset_names --data_dir $lz4_data_path  --output_dir $shm_dir/zarr
echo "Data extracted"

# For multi-dataset
command="$accelerate_path launch --num_processes $SLURM_GPUS \
    train.py --config-name=train_diffusion_unet_timm_umi_workspace_new_dataloader \
    task/dataset=umi_multi_dataset \
    task.dataset.dataset_root_dir=$shm_dir/zarr \
    run_name=$run_name" 

    # task.dataset.used_episode_indices_file=$local_dataset_dir/meta/sampled_500_index_3_datasets.json \
# For the new lazy dataset
# command="$accelerate_path launch --num_processes 8 train.py --config-name=train_diffusion_unet_timm_umi_workspace_new_dataloader task.dataset.zarr_path=/dev/shm/uva/umi_data/cup_arrangement_0.zarr"

# For the original dataset
# command="$accelerate_path launch --num_processes 4 train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=$shm_dir/cup_arrangement_0.zarr"
echo "Running command:"
echo $command
echo "--------------------------------"
exec $command

# rm -rf $shm_dir 
# echo "Shared memory directory cleaned"
