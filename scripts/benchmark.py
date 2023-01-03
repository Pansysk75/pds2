import subprocess
import itertools
import argparse
import re
import csv
import numpy as np
import math


results_file_name = "results.csv"

base_command = "mpirun -np 4 bin/mpi"
dataset = "datasets/mnist_test.csv"

n_list = list(range(100, 1000, 100))

# Or, uncomment for log-spaced numbers! :

# n_start, n_end, n_num_points = (100, 10000)
# n_list = [int(n) for n in np.logspace(
#         math.log10(n_start), math.log10(n_end), int(n_num_points))]

d_list = [100]
k_list = [100]

with open(results_file_name, "w", newline="") as csvfile:
    # create a CSV writer
    writer = csv.writer(csvfile)
    # write the header row
    header = ["base_command", "dataset", "n", "d", "k", "time"]
    writer.writerow(header)

    # iterate over all combinations of the ranges
    for combination in itertools.product(n_list, d_list, k_list):
        n, d, k = combination
        if(n<=k):
            continue
        # build the argument list for the subprocess call
        exec_args = []
        exec_args.extend(base_command.split(" "))
        # add the static arguments
        exec_args.extend("{} {} {} {}".format(dataset, n, d, k).split(" "))

        # run the executable and capture the output
        print("Running command: " + " ".join(exec_args))
        result = subprocess.run(exec_args, stdout=subprocess.PIPE)

    
        output = result.stdout.decode("utf-8")
        # extract the captures from the output
        match = re.search("Total time: (\d+\.?\d*)", output)
        captured = ""
        if match:
            captured = match.group(1)

        # write the results to the CSV file
        row = [base_command, dataset]
        row.extend(combination)
        row.append(captured)
        writer.writerow(row)
