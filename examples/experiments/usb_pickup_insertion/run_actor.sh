export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/plug3 \
    --wandb_mode=offline \
    --wandb_output_dir=../../experiments/usb_pickup_insertion/wandb \
    --actor \