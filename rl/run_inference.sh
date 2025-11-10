export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python rl/inference.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --checkpoint_path=/home/facelesswei/code/hil-serl/outputs/rlpd_checkpoint/plug_into_socket_with_power_cord \
    --eval_checkpoint_step=136000 \
    --eval_n_trajs=20 \