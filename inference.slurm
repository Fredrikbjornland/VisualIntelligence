#!/usr/bin/bash

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:A100m40:1
#SBATCH --account=ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40000
#SBATCH --job-name="yolov5 rdd"
#SBATCH --output=inference.out
#SBATCH --mail-user=jfbjornl@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}

cd ${WORKDIR}
echo "we are running from this directory: $WORKDIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

# Purge modules
module purge
echo "load anaconda"
module load Anaconda3/2020.07
echo "anaconda loaded"



#Deactivate any spill-over environment from the login node
source deactivate &>/dev/null

conda info
conda info --envs
echo "activate"
source activate condaenv
conda info

# Run training
cd yolov5 && python detect.py --source ../Norway/test/images/ --weights runs/train/yolo_road_damage_detection7/weights/best.pt --conf 0.25 --name yolo_road_damage --save-txt
