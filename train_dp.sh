#!/bin/sh
#SBATCH --job-name=train_dp_cup_bf16
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000073
#SBATCH -G 4
#SBATCH --cpus-per-task=64
#SBATCH --error=outputs/bf16/train_dp.err
#SBATCH --output=outputs/bf16/train_dp.out
working_dir=/scratch/m000073/yihuai/robotics/repositories/policies/imitation-learning-policies/prior_works/universal_manipulation_interface

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75


# root_dir=$(dirname $(realpath $0))
# cd $root_dir
cd $working_dir
python_path=$(conda info --base)/envs/umi/bin/python
accelerate_path=$(conda info --base)/envs/imitation/bin/accelerate

shm_dir=/dev/shm/uva/umi_data

clean_shm() {
    echo "Cleaning shared memory..."
    rm -rf $shm_dir
}

error_handler() {
    echo "Error encountered. Running clean_shm."
    clean_shm
    exit 1
}

# Trap errors (any command failing will call `error_handler`)
trap 'error_handler' EXIT

. $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
conda activate umi

$python_path scripts/extract_umi_data.py cup_arrangement_0
command="$accelerate_path launch --num_processes 4 train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset.zarr_path=/dev/shm/uva/umi_data/cup_arrangement_0.zarr"
echo $command
exec $command
