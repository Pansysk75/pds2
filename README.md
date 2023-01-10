# Brute-force kNN in a distributed system using MPI

2nd Assignment for 2022-2023 Parallel and Distributed Systems course, Aristotle University of Thessaloniki 

## Requirements:  
- Recent MPI installation (must support C++17)
- OpenBLAS

## Build:  
Run ```make``` in the top directory. The executables will appear in folder ```bin/```.

## Run:  
There are four executables:
- mpi: The main executable that runs the distributed kNN algorithm.
- test_impl: Runs a test kNN run on a single process.
- test_mpi: Runs a test data transmission between MPI processes.
- generate_dataset: Used to generate a dataset (can be ignored).

The executables will provide information about additional arguments that are needed. For the main "mpi" executable, you may provide the included dataset located in ```datasets/``` folder.
