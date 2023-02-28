#!/bin/bash -x

#SBATCH -n 2                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p serial_requeue
#SBATCH -t 04:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=12000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o joblogs/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joblogs/%A_%a.err  # File to which STDERR will be written, %j inserts jobid

set -x

date
cdir=$(pwd)
tempdir="/scratch/jkillian/${SLURM_JOB_ID}/"
mkdir -p $tempdir
echo $tempdir
cd $tempdir


TREATMENT_LENGTH=40
datatype='healthcare'
SAVENAME="healthcare_2_19"
FILE_ROOT="$SCRATCH/tambe_lab/Users/jkillian/multi_action_bandits"


python3 ${cdir}/adherence_simulation.py -pc ${1} -l ${TREATMENT_LENGTH} -d ${datatype} -s 0 -ws 0 -sv ${SAVENAME} -sid ${SLURM_ARRAY_TASK_ID} --file_root ${FILE_ROOT}

