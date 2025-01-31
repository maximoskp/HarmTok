#!/bin/bash

# List of Python scripts with their respective arguments

# scripts=(
#     "generate_gpt.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
#     "generate_gpt.py -t RootTypeTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
#     "generate_gpt.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
#     "generate_bart.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
#     "generate_bart.py -t RootTypeTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
#     "generate_bart.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
# )

# scripts=(
#     "generate_gpt.py -t RootPCTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
#     "generate_bart.py -t RootPCTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
# )

scripts=(
    "generate_gpt.py -t RootPCTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g -1 -b 5"
)

# Name of the conda environment
conda_env="torch"

# Loop through the scripts and create a screen for each
for script in "${scripts[@]}"; do
    # Extract the base name of the script (first word) to use as the screen name
    screen_name=$(basename "$(echo $script | awk '{print $1}')" .py)
    
    # Start a new detached screen and execute commands
    screen -dmS "$screen_name" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh;  # Update this path if your conda is located elsewhere
        conda activate $conda_env;
        python $script;
        exec bash
    "
    echo "Started screen '$screen_name' for script '$script'."
done
