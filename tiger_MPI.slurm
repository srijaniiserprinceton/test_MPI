#!/bin/bash
#SBATCH --job-name=MCMC_MPI_eigprob      # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks=80              # number of tasks
#sbatch --ntasks-per-node=40     # number of tasks per node
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=60G
# SBATCH --mem-per-cpu=18G        # memory per cpu-core (4G is default)
#SBATCH --time=00:04:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sbdas@princeton.edu

module purge
module load anaconda3
module load openmpi/gcc
conda activate helio
mpiexec -n 80 python test_emcee_MPI.py
