#!/usr/bin/env bash

export PYTHONPATH="/data1/is156025/zb276885/tango/:$PYTHONPATH"

# Set bash to 'debug' mode: exit on error, undefined variables, pipeline errors.
set -e
set -u
set -o pipefail

# Logging function
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Timer
SECONDS=0

#####################
# Default parameters
#####################
stage=1               # Start at stage 1.
stop_stage=10000      # Stop at stage 10000 (i.e. run all stages unless overridden).

# Directories (SET THESE VARIABLES)
DATASET_DIR="/data1/is156025/zb276885/Datasets/LibriHAid/data/mix"  # Raw dataset
DATA_DIR="/database2/LibriHAid/LMFCA-Net" # STFT and oracles are stored here
FILELIST_DIR="/data1/is156025/zb276885/LMFCA-Net/data" # Filelists are store here


# Python script paths - from Tango
TRANSFORM_SCRIPT="/data1/is156025/zb276885/tango/tango/training/transform_dataset.py"
PREPROCESS_SCRIPT="/data1/is156025/zb276885/LMFCA-Net/tools/prepocess_data.py"

# New adapted for LMFCA-Net
ORACLE_SCRIPT="/data1/is156025/zb276885/LMFCA-Net/tools/get_cIRM.py"
CREATE_LISTS_SCRIPT="/data1/is156025/zb276885/LMFCA-Net/tools/create_file_lists.py"
TRAIN_SCRIPT="/data1/is156025/zb276885/LMFCA-Net/train.py"

# Training configuration
CONFIG_YAML="/data1/is156025/zb276885/LMFCA-Net/configs/default.yaml"
GPU="0"
# show GPU status before training
SHOW_NVIDIA_SMI=true

# Source parse_options.sh to handle command-line arguments
. parse_options.sh || echo "No parse_options.sh available, using defaults."


###########################################
# Step 1: Transform Dataset
###########################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: Transforming dataset (copying and renaming files)..."
  python3 "$TRANSFORM_SCRIPT" --dataset_root "$DATASET_DIR" --transformed_data_root "$DATA_DIR"
fi

###########################################
# Step 2: Preprocessing
###########################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: Preprocessing .wav files to compute STFT features ..."
  python3 "$PREPROCESS_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --config_file "$CONFIG_YAML" 
fi

###########################################
# Step 3: Get Ideal Ratio Masks and Oracles
###########################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Stage 3: Processing .npy STFT files to compute cIRM"
  python3 "$ORACLE_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --config_file "$CONFIG_YAML" 
fi


###########################################
# Step 4: Create File Lists
###########################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Stage 4: Creating file lists for training..."
  python3 "$CREATE_LISTS_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$FILELIST_DIR" \
    --config_file "$CONFIG_YAML"
fi

###########################################
# Step 5: Training
###########################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
    log "Using GPU index ${GPU} -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    if [ "${SHOW_NVIDIA_SMI}" = true ] && command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || true
    fi
  log "Stage 5: Running the training pipeline using configuration from YAML file..."
  
  python3 "$TRAIN_SCRIPT" \
    --files_to_load "$FILELIST_DIR" \
    --config_yaml "$CONFIG_YAML" 
fi

###########################################
# Final message
###########################################
log "All steps completed. Total elapsed time: ${SECONDS}s"
