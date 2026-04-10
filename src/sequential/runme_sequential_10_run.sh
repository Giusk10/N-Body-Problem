#!/bin/bash
PROGRAM_NAME="sequential"
SRC="sequential_nBody.c"

PARTICLES_LIST=$(seq 1000 1000 10000)
NUM_RUNS=10

extract_avg()       { grep "Avg iteration time:" "$1" | awk '{print $4}'; }
extract_particles() { grep "Number of particles" "$1" | awk '{print $4}'; }

# Compile
gcc -o "$PROGRAM_NAME" "$SRC" -lm || { echo "Errore compilazione"; exit 1; }

FINAL_FILE="${PROGRAM_NAME}_results.txt"
> "$FINAL_FILE"

for particles in $PARTICLES_LIST; do
    particles_file="merge_${PROGRAM_NAME}_${particles}.txt"
    echo "# version=$PROGRAM_NAME particles=$particles  avg_iteration_time particles" > "$particles_file"

    # 1) 10 run per (particles)
    run_files=()
    for r in $(seq 1 $NUM_RUNS); do
        rf="run_${PROGRAM_NAME}_${particles}_${r}.txt"
        echo "=== $PROGRAM_NAME: $particles particles, run $r ==="
        ./"$PROGRAM_NAME" "$particles" > "$rf"
        run_files+=("$rf")
    done

    # 2) Merge dei 10 run
    merge_file="merge_${PROGRAM_NAME}_${particles}_runs.txt"
    {
        echo "# avg_iteration_time particles (10 runs)"
        for rf in "${run_files[@]}"; do
            avg=$(extract_avg       "$rf")
            p=$(  extract_particles "$rf")
            echo "$avg $p"
        done
    } > "$merge_file"

    # 3) Elimina i 10 file di run
    rm -f "${run_files[@]}"

    # 4) Appendi il merge al file del taglio particelle
    tail -n +2 "$merge_file" >> "$particles_file"
    rm -f "$merge_file"

    # 5) Appendi il file del taglio particelle al file definitivo
    cat "$particles_file" >> "$FINAL_FILE"
    echo "" >> "$FINAL_FILE"

    # 6) Elimina il file del taglio particelle
    rm -f "$particles_file"
done

rm -f "$PROGRAM_NAME"
echo "Completato: $FINAL_FILE"
