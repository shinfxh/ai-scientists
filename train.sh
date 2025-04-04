#!/bin/bash

# Source environment variables
source /etc/profile

# Load the required Anaconda module
module load anaconda/Python-ML-2023b

# Print the value of LLSUB_RANK (if set)
echo "LLSUB_RANK is: $LLSUB_RANK"

# Correct variable assignment
file="continued.py"

# Run the Python script with the given arguments
python "$file" "$LLSUB_RANK" 7


# python train.py $LLSUB_RANK 5 1

# python train.py $LLSUB_RANK 6 3