# Files Exaplaination

- `generate_dataset`: generates a `.csv` file with input points for the algorithm. Can be either points in a *random* or a *regular* grid
- `mpi`: main executable, runs the knn algorithm with specified parameters distributed over *n* processes using _MPI_
- `test_impl`: parametric test of different implementations without utilizing _MPI_
- `test_mpi`: test of communication between different _MPI_ procceses


- `detail/`
    - `communication`: Handles communication between mpi processes
    - `fileio`: Handles loading and saving `.csv` files
    - `globals`: variables accessable from all areas of the program
    - `knn_algorithms`: various implementations of the knn algorithm for a chunk of work
    - `knn_structs`: The data structures used to reason about input and output data
    - `knn_utils`: ambiguous helper functions used in the knn_algorithms
    - `mpi_process`: MPI high level wrapper
    - `utilities`: printing and logging functions
    - `worker`: the unit of work execution for each process