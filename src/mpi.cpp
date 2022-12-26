#include <iostream>
#include <string>
#include <vector>

#include <mpi/mpi.h>

#include "misc.hpp"

#include "mpi_process.hpp"
#include "communication.hpp"
#include "points_chunk.hpp"
#include "worker.hpp"

points_chunk import_data(int idx_start, int idx_end){
    // Imitates importing data
    int size = idx_end - idx_start;
    std::vector<double> vec(size);
    std::iota(vec.begin(), vec.end(), idx_start);
    return {std::move(vec), size, idx_start, idx_end};
}

//Entry point for MPI master
void master_main(mpi_process& process){

    com_port com(process.world_rank, process.world_size);

    int part_size = 31250000; 
    // 31250000 doubles = 250Mb
    // each process will allocate 250*3 = 750 Mb

    // Distribute some work
    initial_work_data init_data{part_size,2,3};
    for(int i=1; i<process.world_size; i++){
        int idx_start = i*part_size;
        int idx_end = (i+1)*part_size;
        points_chunk points = import_data(idx_start, idx_end);
        com.send(init_data, i);
        com.send(points, i);
        com.send(points, i);
    }
    // Give work to local worker
    worker w(process.world_rank, process.world_size, true);

    points_chunk points = import_data(0, part_size);
    w.init_data = init_data;
    w.corpus_set = points;
    w.query_set = std::move(points);
    w.receiving_set = points_chunk(part_size, 0, 0);

    w.work();

    // TODO: gather result

}

void slave_main(mpi_process& process){

    worker w(process.world_rank, process.world_size);
    w.work();

}



// MPI entry:
int main(){

    mpi_process process;

    if (process.world_rank==0) {

        master_main(process);
     
    }
    else{

        slave_main(process);

    }

    return 0;
}