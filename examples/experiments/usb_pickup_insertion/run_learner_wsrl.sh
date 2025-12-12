# /home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-28/usb_pickup_insertion_30_16-47-00.pkl
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_wsrl.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --pretrained_checkpoint_path=/home/facelesswei/code/hil-serl-zbh/examples/experiments/usb_pickup_insertion/plug_with_power_cord_pretrain32/checkpoint_175000 \
    --checkpoint_path=../../experiments/usb_pickup_insertion/wsrl_plug_into_socket_with_power_cord2 \
    --demo_path=/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-12-10/plug_into_socket_with_power_cord_wsrl_227_10-40-00.pkl \
    --wandb_mode=offline \
    --wandb_output_dir=../../experiments/usb_pickup_insertion/wandb \
    --learner \ 