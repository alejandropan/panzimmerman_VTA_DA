#!/bin/bash
# One SLURM array job for one decoder. The spec name arrives in $SPEC:
#
#   sbatch --export=ALL,SPEC=vcs --array=0-<counter-1> decoder_job.cmd
#
# `launch_decoders.py` builds that line for you, including the array size.
# This replaces thirteen .cmd files that differed only in the script path.
#
#SBATCH --job-name=decoder
#SBATCH --partition=witten          # PLACEHOLDER - set to your partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=PLACEHOLDER@princeton.edu

module load anacondapy
source activate iblenv

if [ -z "$SPEC" ]; then
    echo "SPEC is not set. Submit with: sbatch --export=ALL,SPEC=<spec name> ..."
    exit 1
fi

CODE_DIR=/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster
python "$CODE_DIR/run_decoder_job.py" "$SPEC" "$SLURM_ARRAY_TASK_ID"
