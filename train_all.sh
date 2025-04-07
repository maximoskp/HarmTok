#!/bin/bash

# List of Python scripts with their respective arguments

scripts=(
    "train_mlm.py -t ChordSymbolTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 50 -l 5e-5 -b 6"
    "train_mlm.py -t RootTypeTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 50 -l 5e-5 -b 6"
    "train_mlm.py -t PitchClassTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 50 -l 5e-5 -b 6"
    "train_mlm.py -t RootPCTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 50 -l 5e-5 -b 6"
    "train_gpt.py -t ChordSymbolTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 50 -l 5e-5 -b 6"
    "train_gpt.py -t RootTypeTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 50 -l 5e-5 -b 6"
    "train_gpt.py -t PitchClassTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 50 -l 5e-5 -b 6"
    "train_gpt.py -t RootPCTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 50 -l 5e-5 -b 6"
    "train_bart.py -t ChordSymbolTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 50 -l 5e-5 -b 6"
    "train_bart.py -t RootTypeTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 50 -l 5e-5 -b 6"
    "train_bart.py -t PitchClassTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 50 -l 5e-5 -b 6"
    "train_bart.py -t RootPCTokenizer -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 50 -l 5e-5 -b 6"
)

# scripts=(
#     "train_mlm.py -t ChordSymbolTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 0 -e 10 -l 5e-5 -b 6"
#     "train_mlm.py -t RootTypeTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 1 -e 10 -l 5e-5 -b 6"
#     "train_mlm.py -t PitchClassTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 2 -e 10 -l 5e-5 -b 6"
#     "train_mlm.py -t RootPCTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 0 -e 10 -l 5e-5 -b 6"
#     "train_gpt.py -t ChordSymbolTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 0 -e 10 -l 5e-5 -b 6"
#     "train_gpt.py -t RootTypeTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 1 -e 10 -l 5e-5 -b 6"
#     "train_gpt.py -t PitchClassTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 2 -e 10 -l 5e-5 -b 6"
#     "train_gpt.py -t RootPCTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 1 -e 10 -l 5e-5 -b 6"
#     "train_bart.py -t ChordSymbolTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 0 -e 10 -l 5e-5 -b 6"
#     "train_bart.py -t RootTypeTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 1 -e 10 -l 5e-5 -b 6"
#     "train_bart.py -t PitchClassTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 2 -e 10 -l 5e-5 -b 6"
#     "train_bart.py -t RootPCTokenizer -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 2 -e 10 -l 5e-5 -b 6"
# )

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
