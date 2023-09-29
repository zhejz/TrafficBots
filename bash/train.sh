#!/bin/bash
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.out
#SBATCH --time=120:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5000
#SBATCH --tmp=250000
#SBATCH --gpus=rtx_2080_ti:6
#SBATCH --open-mode=truncate

trap "echo sigterm recieved, exiting!" SIGTERM

DATASET_DIR="h5_womd_sim_agent" 
run () {
python -u src/run.py \
model=traffic_bots \
loggers.wandb.entity="YOUR_ENTITY" \
loggers.wandb.name="refactored" \
loggers.wandb.project="trafficbots_release" \
datamodule.data_dir=${TMPDIR}/datasets \
hydra.run.dir='/cluster/scratch/zhejzhan/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

# To resume from a previous run, use
# resume.checkpoint=YOUR_WANDB_RUN_NAME:latest \

source /cluster/project/cvl/zhejzhan/apps/miniconda3/etc/profile.d/conda.sh
conda activate traffic_bots

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`

echo START copying data: `date`
mkdir $TMPDIR/datasets
cp /cluster/scratch/zhejzhan/$DATASET_DIR/training.h5 $TMPDIR/datasets/
cp /cluster/scratch/zhejzhan/$DATASET_DIR/validation.h5 $TMPDIR/datasets/
echo DONE copying: `date`

type run
echo START: `date`
run &
wait
echo DONE: `date`

mkdir -p ./logs/slurm
mv ./logs/$SLURM_JOB_ID.out ./logs/slurm/$SLURM_JOB_ID.out

echo finished at: `date`
exit 0;
