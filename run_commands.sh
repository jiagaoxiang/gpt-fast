#!/bin/bash

# Log file
LOGFILE="output.log"

# Clear the log file if it exists
> $LOGFILE

# Run commands and log output
{
    echo "Running commands..."

    echo -e "\n**************Running baseline with $MODEL_REPO1..."
    python generate.py --checkpoint_path checkpoints/$MODEL_REPO1/model.pth --prompt "Hello, my name is"
    echo -e "\n**************Running torch.compile with $MODEL_REPO1..."
    python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO1/model.pth --prompt "Hello, my name is"

    echo "Setting DEVICE to cuda..."
    export DEVICE=cuda

    echo -e "\n**************Quantizing and running commands with $MODEL_REPO1..."
    python quantize.py --checkpoint_path checkpoints/$MODEL_REPO1/model.pth --mode int8
    echo -e "\n**************Running int8 with $MODEL_REPO1..."
    python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO1/model_int8.pth --device $DEVICE

    echo -e "\n**************Running baseline with $MODEL_REPO2..."
    python generate.py --checkpoint_path checkpoints/$MODEL_REPO2/model.pth --prompt "Hello, my name is"
    echo -e "\n**************Running torch.compile with $MODEL_REPO2..."
    python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO2/model.pth --prompt "Hello, my name is"

    echo "Quantizing and running commands with $MODEL_REPO2..."
    python quantize.py --checkpoint_path checkpoints/$MODEL_REPO2/model.pth --mode int8
    echo -e "\n**************Running int8 with $MODEL_REPO2..."
    python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO2/model_int8.pth --device $DEVICE

    echo -e "\n**************Running baseline with $MODEL_REPO3..."
    python generate.py --checkpoint_path checkpoints/$MODEL_REPO3/model.pth --prompt "Hello, my name is"
    echo -e "\n**************Running torch.compile with $MODEL_REPO3..."
    python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO3/model.pth --prompt "Hello, my name is"

    echo "Quantizing and running commands with $MODEL_REPO3..."
    python quantize.py --checkpoint_path checkpoints/$MODEL_REPO3/model.pth --mode int8
    echo "/n**************Running int8 with $MODEL_REPO3..."
    python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO3/model_int8.pth --device $DEVICE

} &> $LOGFILE
