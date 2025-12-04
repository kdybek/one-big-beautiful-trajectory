#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=experiment_out.txt

ml ML-bundle/24.06a

cd $SCRATCH/one-big-beautiful-trajectory
python -m venv .venv
source .venv/bin/activate

wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O bazel
chmod +x bazel
export PATH="$PWD:$PATH"

pip install -r ~/one-big-beautiful-trajectory/requirements.txt

rm bazel
