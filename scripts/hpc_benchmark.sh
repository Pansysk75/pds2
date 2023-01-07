#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=01:00:00

### Above are global default values, stuff like "--nodes", "--tasks"
### are set in the python script that actually launches the jobs

## Load modules
module load gcc/10.2.0 python openblas openmpi pkgconf

### Install python virtual environment and modules
python -m venv scripts/python_venv
source scripts/python_venv/bin/activate
pip install -r scripts/requirements.txt

### Clean / compile program
make clean
make BUILD_ENV=hpc

### Run python benchmark script
python scripts/benchmark.py --distributed
