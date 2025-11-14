export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python rl/train.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --checkpoint_path=/home/facelesswei/code/hil-serl-debug/outputs/rlpd/pap_cube \
    --actor \


# --checkpoint_path=/home/facelesswei/code/hil-serl-debug/outputs/rlpd/open_switch \
# --checkpoint_path=/home/facelesswei/code/hil-serl/outputs/rlpd_checkpoint/plug_into_socket_with_power_cord \