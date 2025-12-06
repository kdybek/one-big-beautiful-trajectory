#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=setup_out.txt

ml ML-bundle/24.06a

cd $SCRATCH
mkdir -p one-big-beautiful-trajectory
cd one-big-beautiful-trajectory
cp -ru ~/one-big-beautiful-trajectory/* .
python -m venv .venv
source .venv/bin/activate

wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O bazel
chmod +x bazel
export PATH="$PWD:$PATH"
export USE_BAZEL_VERSION=5.3.2
export XDG_CACHE_HOME=$SCRATCH/.cache

pip install -r requirements.txt

rm bazel
