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
python main.py --env_name={{env}} \
               --seed={{seed}} \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/rpcrl.py \
               --agent.discount={{discount}} \
               --agent.alpha={{alpha}} \
               --agent.beta={{beta1}} \
               --agent.actor_p_randomgoal={{actor_p_randomgoal}} \
               --agent.actor_p_trajgoal={{actor_p_trajgoal}} &

python main.py --env_name={{env}} \
               --seed={{seed}} \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/rpcrl.py \
               --agent.discount={{discount}} \
               --agent.alpha={{alpha}} \
               --agent.beta={{beta2}} \
               --agent.actor_p_randomgoal={{actor_p_randomgoal}} \
               --agent.actor_p_trajgoal={{actor_p_trajgoal}} &

wait

python main.py --env_name={{env}} \
               --seed={{seed}} \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/rpcrl.py \
               --agent.discount={{discount}} \
               --agent.alpha={{alpha}} \
               --agent.beta={{beta3}} \
               --agent.actor_p_randomgoal={{actor_p_randomgoal}} \
               --agent.actor_p_trajgoal={{actor_p_trajgoal}} &

python main.py --env_name={{env}} \
               --seed={{seed}} \
               --run_group=beta_search \
               --eval_episodes=50 \
               --agent=agents/crl.py \
               --agent.discount={{discount}} \
               --agent.alpha={{alpha}} \
               --agent.actor_p_randomgoal={{actor_p_randomgoal}} \
               --agent.actor_p_trajgoal={{actor_p_trajgoal}} &

wait
