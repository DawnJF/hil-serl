export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=iphone_charging \
    --checkpoint_path=../../experiments/usb_pickup_insertion/iphone \
    --wandb_mode=online \
    --wandb_output_dir=../../experiments/usb_pickup_insertion/wandb \
    --actor \