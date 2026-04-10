#!/bin/bash

# MPI script for running all three parallel versions with different numbers of processes and particles

# Compile all three versions
mpicc -o mpi            mpi_nBody.c            -lm
mpicc -o mpi_allgatherv mpi_nBody_allgatherv.c -lm
mpicc -o mpi_ring       mpi_nBody_ring.c       -lm

# Output files (one per version)
output_original="mpi_results.txt"
output_allgatherv="mpi_allgatherv_results.txt"
output_ring="mpi_ring_results.txt"

# Clear previous result files
> "$output_original"
> "$output_allgatherv"
> "$output_ring"

# Loop over different numbers of particles
for num_particles in {1000..10000..1000}
do
    # Loop over different numbers of processes
    for num_processes in 1 2 3 4
    do
        echo "=== Original (Bcast+Gatherv): $num_processes processes, $num_particles particles ==="
        mpirun -np $num_processes ./mpi $num_particles >> "$output_original"
        echo "--------------------------------------------------------"

        echo "=== Allgatherv: $num_processes processes, $num_particles particles ==="
        mpirun -np $num_processes ./mpi_allgatherv $num_particles >> "$output_allgatherv"
        echo "--------------------------------------------------------"

        echo "=== Ring (Sendrecv): $num_processes processes, $num_particles particles ==="
        mpirun -np $num_processes ./mpi_ring $num_particles >> "$output_ring"
        echo "--------------------------------------------------------"
    done
done

# Clean up compiled binaries
rm -f mpi mpi_allgatherv mpi_ring
