export http_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
export https_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
apt update
apt install -y libgl1 libglib2.0-0

NAME=pi05_calvin_xz-freezevision-128
CODE_DIR=/mnt/afs/fanxianzhe/openpi

GLOBAL_BATCH_SIZE=128
NUM_GPUS=$(ls /dev | grep -E 'nvidia[0-9]+' | wc -l)
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=
export WANDB_API_KEY=
export WANDB__SERVICE_WAIT=3600
export WANDB_INIT_TIMEOUT=3600
export WANDB_MODE=online
export PYTHONPATH=$PYTHONPATH:/mnt/afs/fanxianzhe/openpi
export JAX_LOG_COMPILES=1

[ -d /root ] && mv /root /root.bak
ln -s /mnt/afs/fanxianzhe /root
source /root/.bashrc
# conda create -n openpi_env -c conda-forge python=3.10 -y
conda activate openpi_env
# conda install -c conda-forge ffmpeg=7 av=14.4.0 -y
# conda install -c conda-forge pkg-config cython compilers -y
# pip install -e /mnt/afs/fanxianzhe/openpi

cd $CODE_DIR
XLA_PYTHON_CLIENT_PREALLOCATE=false

export XDG_CACHE_HOME=/tmp/xcache
export HF_HOME=/tmp/hf
export HF_HUB_CACHE=/tmp/hf/hub
export HUGGINGFACE_HUB_CACHE=/tmp/hf/hub
export TRANSFORMERS_CACHE=/tmp/hf/transformers

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

mkdir -p /tmp/xcache /tmp/hf/hub /tmp/hf/transformers /tmp/wandb /tmp/jax_cache

uv run scripts/train.py pi05_calvin_xz --exp-name=$NAME --batch-size=$GLOBAL_BATCH_SIZE --fsdp-devices=$NUM_GPUS --save-interval 500
sleep inf