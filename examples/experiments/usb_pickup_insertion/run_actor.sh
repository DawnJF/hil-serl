export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
sudo /home/robot/miniforge3/envs/roboar4il/hilserl/bin/python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    # --checkpoint_path=../../experiments/usb_pickup_insertion/debug \
    --actor \