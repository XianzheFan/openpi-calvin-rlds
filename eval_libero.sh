source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

TASK_SUITE=$1
SAVE_NAME=$2

python examples/libero/main.py --args.task_suite_name=${TASK_SUITE} --args.save_name=${SAVE_NAME}