#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
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

# OBBT
python main.py --env_name=humanoidmaze-medium-stitch-v0 \
               --seed=2 \
               --run_group=model_size_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.model_size_testing=True \
               --agent.hidden_dim_size=512 \
               --agent.num_hidden_layers=6 \
               --agent.per_traj_samples=32 \
               --agent.discount=0.995 \
               --agent.alpha=0.1 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 \
               --agent.dataset_class=OBBTDataset &

python main.py --env_name=humanoidmaze-medium-stitch-v0 \
               --seed=2 \
               --run_group=model_size_search  \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.model_size_testing=True \
               --agent.hidden_dim_size=512 \
               --agent.num_hidden_layers=9 \
               --agent.per_traj_samples=32 \
               --agent.discount=0.995 \
               --agent.alpha=0.1 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 \
               --agent.dataset_class=OBBTDataset &

wait

python main.py --env_name=humanoidmaze-medium-stitch-v0 \
               --seed=2 \
               --run_group=model_size_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.model_size_testing=True \
               --agent.hidden_dim_size=512 \
               --agent.num_hidden_layers=6 \
               --agent.per_traj_samples=512 \
               --agent.discount=0.995 \
               --agent.alpha=0.1 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 \
               --agent.dataset_class=OBBTDataset &

python main.py --env_name=humanoidmaze-medium-stitch-v0 \
               --seed=2 \
               --run_group=model_size_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.model_size_testing=True \
               --agent.hidden_dim_size=512 \
               --agent.num_hidden_layers=9 \
               --agent.per_traj_samples=512 \
               --agent.discount=0.995 \
               --agent.alpha=0.1 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 \
               --agent.dataset_class=OBBTDataset &

wait

# Standard CRL
python main.py --env_name=humanoidmaze-medium-stitch-v0 \
               --seed=2 \
               --run_group=model_size_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.model_size_testing=True \
               --agent.hidden_dim_size=512 \
               --agent.num_hidden_layers=6 \
               --agent.discount=0.995 \
               --agent.alpha=0.1 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 &

python main.py --env_name=humanoidmaze-medium-stitch-v0 \
               --seed=2 \
               --run_group=model_size_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.batch_size=1024 \
               --agent.model_size_testing=True \
               --agent.hidden_dim_size=512 \
               --agent.num_hidden_layers=9 \
               --agent.discount=0.995 \
               --agent.alpha=0.1 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 &

wait
