# 6  255000
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../inference.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --checkpoint_path=/home/facelesswei/code/hil-serl-zbh/examples/experiments/usb_pickup_insertion/plug_with_power_cord_pretrain40 \
    --eval_checkpoint_step=315000 \
    --eval_n_trajs=70 \

