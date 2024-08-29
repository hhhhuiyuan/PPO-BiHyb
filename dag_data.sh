#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=50
#submit by $sbatch dag_data.sh
source activate Dag

python DIFF_dag_data.py --config DIFF_dag.yaml