export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \

python ../../train_iql.py "$@" \
    --exp_name=plug_into_socket_with_power_cord \
    --iql_checkpoint_path=../../experiments/usb_pickup_insertion/plug_with_power_cord_pretrain40 \
    --demo_path=/ssdwork/liujinxin/DATASET/Jax_Hil_Serl_Dataset/2025-11-06/plug_into_socket_with_power_cord_433_with_demoBuffer_10-25-00.pkl \
    --save_period 1000 \
    --train_steps 500000 \
    --use_calql False \
    --reward_scale 1.0 \
    --reward_bias 0.0 \
    --wandb_mode=offline \
    --wandb_output_dir=../../experiments/usb_pickup_insertion/wandb \