# 6  255000
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../inference.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --checkpoint_path=/home/facelesswei/code/hil-serl-zbh-wsrl/examples/experiments/usb_pickup_insertion/wsrl_plug_into_socket_with_power_cord2 \
    --eval_checkpoint_step=52000 \
    --eval_n_trajs=70 \
    --use_calql=False

