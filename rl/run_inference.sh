export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python rl/inference.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --checkpoint_path=/home/facelesswei/code/hil-serl-debug/outputs/rlpd/pap_cube \
    --eval_checkpoint_step=19600 \
    --eval_n_trajs=20 \



# plug_p 11000