export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../inference.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/plug3 \
    --eval_checkpoint_step=27000 \
    --eval_n_trajs=2 \