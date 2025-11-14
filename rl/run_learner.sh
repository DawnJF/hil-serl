export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python rl/train.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --wandb_name=pap_cube \
    --checkpoint_path=/home/facelesswei/code/hil-serl-debug/outputs/rlpd/pap_cube \
    --learner \


    # --demo_path=/home/facelesswei/code/hil-serl/datasets/trajectories/2025-10-29/merged/merged_data.pkl \
    # --checkpoint_path=/home/facelesswei/code/hil-serl-debug/outputs/rlpd/open_switch \
# --demo_path=/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-10-27/usb_pickup_insertion_31_18-18-00.pkl \
# --checkpoint_path=/home/facelesswei/code/hil-serl/outputs/rlpd_checkpoint/plug_into_socket_with_power_cord \