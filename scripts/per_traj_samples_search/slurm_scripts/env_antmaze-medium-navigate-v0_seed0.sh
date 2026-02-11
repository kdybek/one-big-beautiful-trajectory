#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

cd $SCRATCH/one-big-beautiful-trajectory
cp -ru ~/one-big-beautiful-trajectory/* .
source .venv/bin/activate

python main.py --env_name=antmaze-medium-navigate-v0 \
               --seed=0 \
               --run_group=per_traj_samples_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.per_traj_samples=32 \
               --agent.alpha=0.03 \
               --agent.dataset_class=OBBTDataset &

python main.py --env_name=antmaze-medium-navigate-v0 \
               --seed=0 \
               --run_group=per_traj_samples_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.per_traj_samples=128 \
               --agent.alpha=0.03 \
               --agent.dataset_class=OBBTDataset &

python main.py --env_name=antmaze-medium-navigate-v0 \
               --seed=0 \
               --run_group=per_traj_samples_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.per_traj_samples=512 \
               --agent.alpha=0.03 \
               --agent.dataset_class=OBBTDataset &

python main.py --env_name=antmaze-medium-navigate-v0 \
               --seed=0 \
               --run_group=per_traj_samples_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.alpha=0.03 &

wait
