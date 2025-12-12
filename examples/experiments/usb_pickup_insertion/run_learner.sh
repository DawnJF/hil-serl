# /home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-28/usb_pickup_insertion_30_16-47-00.pkl
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=iphone_charging \
    --checkpoint_path=../../experiments/usb_pickup_insertion/iphone \
    --demo_path=/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-11-05/iphone_charging_20_15-00-20.pkl \
    --wandb_mode=online \
    --wandb_output_dir=../../experiments/usb_pickup_insertion/wandb \
    --learner \ 