#!/bin/bash
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.out
#SBATCH --time=24:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15000
#SBATCH --tmp=250000
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --open-mode=truncate

trap "echo sigterm recieved, exiting!" SIGTERM

DATASET_DIR="h5_womd_sim_agent" 
run () {
python -u src/run.py \
resume=submission \
action=validate \
trainer.limit_val_batches=1.0 \
resume.checkpoint=YOUR_WANDB_RUN_NAME:latest \
loggers.wandb.entity="YOUR_ENTITY" \
loggers.wandb.name="womd_validate" \
loggers.wandb.project="trafficbots_sub" \
datamodule.data_dir=${TMPDIR}/datasets \
hydra.run.dir='/cluster/scratch/zhejzhan/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

# for testing
# run () {
# python -u src/run.py \
# resume=submission \
# action=test \
# trainer.limit_test_batches=1.0 \
# resume.checkpoint=YOUR_WANDB_RUN_NAME:latest \
# loggers.wandb.entity="YOUR_ENTITY" \
# loggers.wandb.name="womd_test" \
# loggers.wandb.project="trafficbots_sub" \
# datamodule.data_dir=${TMPDIR}/datasets \
# hydra.run.dir='/cluster/scratch/zhejzhan/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
# }


source /cluster/project/cvl/zhejzhan/apps/miniconda3/etc/profile.d/conda.sh
conda activate traffic_bots

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`

echo START copying data: `date`
mkdir $TMPDIR/datasets
cp /cluster/scratch/zhejzhan/$DATASET_DIR/validation.h5 $TMPDIR/datasets/
cp /cluster/scratch/zhejzhan/$DATASET_DIR/testing.h5 $TMPDIR/datasets/
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
