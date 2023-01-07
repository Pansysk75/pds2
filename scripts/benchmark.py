import subprocess
import itertools
import argparse
import re
import csv
import numpy as np

### Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--distributed", help="Use this to launch slurm jobs", action="store_true")
args = vars(parser.parse_args())


### Constant run parameters

results_file_name = "results.csv"

base_command = "mpirun -np {} bin/mpi -f={} -n={} -d={} -k={}"
if(args["distributed"]):
    base_command = "srun --nodes {} bin/mpi -f={} -n={} -d={} -k={}"

dataset = "datasets/mnist_test.csv"


### Below we set the variables that run parametrically
### Here are some examples:

### This is for manually specifying values:
# n_list = [100, 200, 400, 525, 1000]

### This is for linear range
# n_list = list(range(200, 2000, 200))

### This is for a list of powers of 2 ([2**7, 2**8, ..., 2**14])
# n_list = [2**i for i in range(7, 14)]

### This is an alternative for log-spaced integers
# n_start, n_end, n_num_points = (100, 10000, 10)
# n_list = [int(n) for n in np.logspace(math.log10(n_start), math.log10(n_end), int(n_num_points))]

# n_list = [2**i for i in range(7, 14)]
n_list = [2000]
d_list = [8,16,32, 64, 128]
k_list = [5]
mpi_np_list = [1,4]

### Repeat each measurement many times
n_iterations = 5 

with open(results_file_name, "w", newline="") as csvfile:
    ### Create a CSV writer
    writer = csv.writer(csvfile)
    ### Write the header row
    header = ["base_command", "dataset", "num_processors", "n", "d", "k", "time"]
    writer.writerow(header)

    ### Iterate over all combinations of the ranges
    for combination in itertools.product(mpi_np_list, n_list, d_list, k_list):
        mpi_np, n, d, k = combination
        if(n<=k):
            continue
        ### Build the argument list for the subprocess call
        exec_args = base_command.format(mpi_np, dataset, n, d, k).split(" ")

        print("Running command: " + " ".join(exec_args))
        for _ in range(n_iterations):
            ### Run the executable and capture the output
            result = subprocess.run(exec_args, stdout=subprocess.PIPE)

        
            output = result.stdout.decode("utf-8")
            ### Extract the captures from the output
            match = re.search("Total time: (\d+\.?\d*)", output)
            captured = ""
            if match:
                captured = match.group(1)

            ### Wprite the results to the CSV file
            row = [base_command, dataset, *combination, captured]
            writer.writerow(row)
