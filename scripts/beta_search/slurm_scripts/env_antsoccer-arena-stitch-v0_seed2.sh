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

# RPCRL
python main.py --env_name=antsoccer-arena-stitch-v0 \
               --seed=2 \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/rpcrl.py \
               --agent.discount=0.99 \
               --agent.alpha=0.3 \
               --agent.beta=0.0 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 &

python main.py --env_name=antsoccer-arena-stitch-v0 \
               --seed=2 \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/rpcrl.py \
               --agent.discount=0.99 \
               --agent.alpha=0.3 \
               --agent.beta=0.01 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 &

wait

python main.py --env_name=antsoccer-arena-stitch-v0 \
               --seed=2 \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/rpcrl.py \
               --agent.discount=0.99 \
               --agent.alpha=0.3 \
               --agent.beta=0.001 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 &

python main.py --env_name=antsoccer-arena-stitch-v0 \
               --seed=2 \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.discount=0.99 \
               --agent.alpha=0.3 \
               --agent.actor_p_randomgoal=0.5 \
               --agent.actor_p_trajgoal=0.5 &

wait
