#!/bin/bash

# Compile
mpicc -std=c99 -o mpi_allgatherv mpi_nBody_allgatherv.c -lm

# Machinefile (must exist in the current directory)
MACHINEFILE="machinefile.txt"

# Output files
output_eth="mpi_allgatherv_results_ethernet.txt"
output_ib="mpi_allgatherv_results_infiniband.txt"

> "$output_eth"
> "$output_ib"

# MCA flags for network selection
ETH_FLAGS="--mca btl self,tcp --mca btl_tcp_if_include em2"
IB_FLAGS="--mca btl self,openib"

# Mapping by-core: fills one node (16 cores) before moving to the next,
# keeping us in shared memory up to 16 processes.
MAP_FLAGS="--map-by core"

for num_particles in $(seq 1000 1000 10000)
do
    for num_processes in 1 2 4 8 16 17 32 33 48 49 64
    do
        echo "=== Allgatherv ETHERNET: $num_processes proc, $num_particles particles ==="
        mpirun $ETH_FLAGS $MAP_FLAGS -np $num_processes \
               -machinefile $MACHINEFILE ./mpi_allgatherv $num_particles >> "$output_eth"
        echo "--------------------------------------------------------"

        echo "=== Allgatherv INFINIBAND: $num_processes proc, $num_particles particles ==="
        mpirun $IB_FLAGS $MAP_FLAGS -np $num_processes \
               -machinefile $MACHINEFILE ./mpi_allgatherv $num_particles >> "$output_ib"
        echo "--------------------------------------------------------"
    done
done

rm -f mpi_allgatherv
