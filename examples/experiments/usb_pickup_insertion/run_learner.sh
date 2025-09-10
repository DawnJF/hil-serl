export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/plug3 \
    --demo_path=/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-09/usb_pickup_insertion_30_11-50-21.pkl \
    --wandb_mode=offline \
    --wandb_output_dir=../../experiments/usb_pickup_insertion/wandb \
    --learner \ 