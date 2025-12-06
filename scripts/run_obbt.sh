#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=experiment_out.txt

ml ML-bundle/24.06a

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

cd $SCRATCH/one-big-beautiful-trajectory
cp -ru ~/one-big-beautiful-trajectory/* .
source .venv/bin/activate

python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.03 --agent.dataset_class=OBBTDataset
