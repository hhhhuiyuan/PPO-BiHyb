#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=50
#SBATCH -A mengdigroup
#SBATCH -p pli

#submit by $sbatch dag_data.sh

module purge
module load anaconda3/2023.9
conda activate DAG

#python DIFF_dag_data.py --config DIFF_dag.yaml
python DIFF_dag_data.py --config DIFF_dag_test.yaml