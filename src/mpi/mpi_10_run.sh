#!/bin/bash
MACHINEFILE="machinefile.txt"
ETH_FLAGS="--mca btl self,tcp --mca btl_tcp_if_include em2"
IB_FLAGS="--mca btl self,openib"
MAP_FLAGS="--map-by core"

PARTICLES_LIST=$(seq 1000 1000 10000)
PROCS_LIST="1 2 4 8 16 17 32 33 48 49 64"
NUM_RUNS=10

VERSIONS="mpi_nBody mpi_nBody_allgatherv mpi_nBody_ring"

extract_avg()       { grep "Avg iteration time:" "$1" | awk '{print $4}'; }
extract_particles() { grep "Number of particles" "$1" | awk '{print $4}'; }
extract_procs()     { grep -iE "Number of (processes|porcesses)" "$1" | awk -F: '{print $2}' | tr -d ' '; }

run_network() {
    local net_name="$1"
    local net_flags="$2"
    local version="$3"
    local binary="./${version}"
    local final_file="${version}_results_${net_name}.txt"
    > "$final_file"

    for particles in $PARTICLES_LIST; do
        particles_file="merge_${version}_${net_name}_${particles}.txt"
        echo "# version=$version particles=$particles  avg_iteration_time particles processes" > "$particles_file"

        for procs in $PROCS_LIST; do
            run_files=()
            for r in $(seq 1 $NUM_RUNS); do
                rf="run_${version}_${net_name}_${particles}_${procs}_${r}.txt"
                echo "=== $version | $net_name: $procs proc, $particles particles, run $r ==="
                mpirun $net_flags $MAP_FLAGS -np $procs \
                       -machinefile $MACHINEFILE $binary $particles > "$rf"
                run_files+=("$rf")
            done

            merge_file="merge_${version}_${net_name}_${particles}_${procs}.txt"
            {
                echo "# avg_iteration_time particles processes (10 runs)"
                for rf in "${run_files[@]}"; do
                    avg=$(extract_avg       "$rf")
                    p=$(  extract_particles "$rf")
                    np=$( extract_procs     "$rf")
                    echo "$avg $p $np"
                done
            } > "$merge_file"

            rm -f "${run_files[@]}"
            tail -n +2 "$merge_file" >> "$particles_file"
            rm -f "$merge_file"
        done

        cat "$particles_file" >> "$final_file"
        echo "" >> "$final_file"
        rm -f "$particles_file"
    done

    echo "Completato: $final_file"
}

for version in $VERSIONS; do
    echo "### Compilazione $version ###"
    mpicc -std=c99 -o "$version" "${version}.c" -lm || { echo "Errore compilazione $version"; exit 1; }

    run_network "ethernet"   "$ETH_FLAGS" "$version"
    run_network "infiniband" "$IB_FLAGS"  "$version"

    rm -f "$version"
done
