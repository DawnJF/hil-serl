export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/debug4 \
    --demo_path=/media/robot/30F73268F87D0FEF/Jax_Hil_Serl_Dataset/2025-09-01/usb_pickup_insertion_50_17-45-18.pkl \
    --wandb_mode=offline \
    --learner \ 