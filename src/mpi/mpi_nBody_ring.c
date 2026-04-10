#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <mpi.h>

#define MASTER 0            // Rank of the MASTER processor
#define I 10                // Number of iterations
#define SOFTENING 1e-9f     // Infinitely large value used in computation

typedef struct {
    float mass;
    float x, y, z;
    float vx, vy, vz;
} Particle;

// Function prototypes
void compute_equal_workload_for_each_task(int *dim_portions, int *displs, int arraysize, int numtasks);
int convertStringToInt(char *str);

int main(int argc, char* argv[]) {
    MPI_Datatype particle_type;             // MPI datatype to communicate the "Particle" data type
    int numtasks;                           // Number of used processors
    int myrank;                             // Rank of the current process
    double start, end, iterStart, iterEnd;  // Variables used for measuring the total execution time and each iteration

    int *dim_portions;                      // Size of the workload portion for each process
    int *displ;                             // Starting offset of the workload portion for each process
    Particle *my_portion;                   // Portion of particles owned by this process

    int num_particles = 1000;  // Default number of particles if no parameter is provided on the command line
    if (argc > 1) {
        // Parameter provided from the command line indicating the number of particles
        // Convert string to integer
        num_particles = convertStringToInt(argv[1]);
    }

    /*** Initialize MPI ***/
    MPI_Init(&argc, &argv);

    /*** Create MPI data type to communicate the "Particle" data type ***/
    MPI_Type_contiguous(7, MPI_FLOAT, &particle_type);
    MPI_Type_commit(&particle_type);

    /*** Get the number of used processors and the rank of the current process ***/
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Barrier(MPI_COMM_WORLD);    /* All processes are initialized */
    start = MPI_Wtime();            /* Record the start time of execution */

    /*** Calculate how particles are evenly associated with the processes ***/
    dim_portions = (int*)malloc(sizeof(int) * numtasks);
    displ = (int*)malloc(sizeof(int) * numtasks);
    compute_equal_workload_for_each_task(dim_portions, displ, num_particles, numtasks);

    const float dt = 0.01f; // Time step

    /* Each process only allocates its own portion — not the full particles array */
    my_portion = (Particle*)malloc(sizeof(Particle) * dim_portions[myrank]);

    /* Find the maximum chunk size across all processes (needed for transit buffers) */
    int max_chunk = 0;
    for (int p = 0; p < numtasks; p++)
        if (dim_portions[p] > max_chunk) max_chunk = dim_portions[p];

    /* Two transit buffers: one holds the current chunk being processed,
       the other receives the next chunk from the ring.
       Their pointers are swapped at each ring step. */
    Particle *buf1 = (Particle*)malloc(sizeof(Particle) * max_chunk);
    Particle *buf2 = (Particle*)malloc(sizeof(Particle) * max_chunk);

    /* Per-particle force accumulators for this process's portion */
    float *Fx = (float*)malloc(sizeof(float) * dim_portions[myrank]);
    float *Fy = (float*)malloc(sizeof(float) * dim_portions[myrank]);
    float *Fz = (float*)malloc(sizeof(float) * dim_portions[myrank]);

    /* Ring neighbors */
    int right = (myrank + 1) % numtasks;
    int left  = (myrank - 1 + numtasks) % numtasks;

    for (int iteration = 0; iteration < I; iteration++) {

        MPI_Barrier(MPI_COMM_WORLD);  // Synchronize processes before starting to measure the iteration execution time
        iterStart = MPI_Wtime();

        if (iteration == 0) {
            /* Only MASTER reads the file, then distributes each process's portion via MPI_Scatterv */
            Particle *all_buf = NULL;
            if (myrank == MASTER) {
                all_buf = (Particle*)malloc(sizeof(Particle) * num_particles);
                FILE *fileRead = fopen("particles.txt", "r");
                if (fileRead == NULL) {
                    printf("\nUnable to open the file.\n");
                    exit(EXIT_FAILURE);
                }

                int particlesRead = fread(all_buf, sizeof(Particle) * num_particles, 1, fileRead);
                if (particlesRead == 0) {
                    printf("ERROR: The number of particles to read is greater than the number of particles in the file\n");
                    exit(EXIT_FAILURE);
                }

                fclose(fileRead);
            }

            MPI_Scatterv(all_buf, dim_portions, displ, particle_type,
                         my_portion, dim_portions[myrank], particle_type,
                         MASTER, MPI_COMM_WORLD);

            if (myrank == MASTER) free(all_buf);
        }
        /* For iteration > 0: my_portion already holds the up-to-date data from the previous iteration
           (each process only updates and retains its own portion — no broadcast needed). */

        /* ------------------------------------------------------------------ *
         *  Ring force computation via MPI_Sendrecv                           *
         *                                                                    *
         *  At each step, every process:                                      *
         *    1. Accumulates forces on my_portion from the current chunk.     *
         *    2. Sends the current chunk to the right neighbor and receives   *
         *       the next chunk from the left neighbor.                       *
         *                                                                    *
         *  After numtasks steps every process has seen all N particles.      *
         * ------------------------------------------------------------------ */

        /* Reset force accumulators */
        for (int i = 0; i < dim_portions[myrank]; i++) {
            Fx[i] = 0.0f; Fy[i] = 0.0f; Fz[i] = 0.0f;
        }

        /* Load own portion into the current transit buffer */
        Particle *cur = buf1;
        Particle *nxt = buf2;
        memcpy(cur, my_portion, sizeof(Particle) * dim_portions[myrank]);
        int chunk_owner = myrank;  // Tracks which process originally owned cur

        for (int step = 0; step < numtasks; step++) {
            int chunk_size = dim_portions[chunk_owner];

            /* Accumulate forces from the current chunk onto each particle in my_portion */
            for (int i = 0; i < dim_portions[myrank]; i++) {
                for (int j = 0; j < chunk_size; j++) {
                    float dx = cur[j].x - my_portion[i].x;
                    float dy = cur[j].y - my_portion[i].y;
                    float dz = cur[j].z - my_portion[i].z;
                    float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    float invDist = 1.0f / sqrtf(distSqr);
                    float invDist3 = invDist * invDist * invDist;

                    Fx[i] += dx * invDist3;
                    Fy[i] += dy * invDist3;
                    Fz[i] += dz * invDist3;
                }
            }

            if (step < numtasks - 1) {
                /* The next chunk to arrive is the one currently held by our left neighbor,
                   which originally belonged to process (chunk_owner - 1). */
                int next_owner = (chunk_owner - 1 + numtasks) % numtasks;
                int recv_size  = dim_portions[next_owner];

                /* Send cur to the right; receive the next chunk from the left */
                MPI_Sendrecv(cur, chunk_size, particle_type, right, 0,
                             nxt, recv_size,  particle_type, left,  0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /* Swap buffer pointers for the next step */
                Particle *tmp = cur; cur = nxt; nxt = tmp;
                chunk_owner = next_owner;
            }
        }

        /* Apply accumulated forces and integrate positions for this process's portion */
        for (int i = 0; i < dim_portions[myrank]; i++) {
            my_portion[i].vx += dt * Fx[i];
            my_portion[i].vy += dt * Fy[i];
            my_portion[i].vz += dt * Fz[i];

            my_portion[i].x += my_portion[i].vx * dt;
            my_portion[i].y += my_portion[i].vy * dt;
            my_portion[i].z += my_portion[i].vz * dt;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        iterEnd = MPI_Wtime();
        if (myrank == MASTER) printf("Iteration %d of %d completed in %f seconds\n", iteration + 1, I, (iterEnd - iterStart));
    }

    MPI_Barrier(MPI_COMM_WORLD);     // All processes have finished
    end = MPI_Wtime();               // Record the end time of execution

    /* Gather the final state of all particles on MASTER for output */
    Particle *all_particles = NULL;
    if (myrank == MASTER) all_particles = (Particle*)malloc(sizeof(Particle) * num_particles);

    MPI_Gatherv(my_portion, dim_portions[myrank], particle_type,
                all_particles, dim_portions, displ, particle_type,
                MASTER, MPI_COMM_WORLD);

    MPI_Type_free(&particle_type);
    MPI_Finalize();

    if (myrank == MASTER) {
        double totalTime = end - start;
        double avgTime = totalTime / (double)(I);
        printf("\nAvg iteration time: %f seconds\n", avgTime);
        printf("Total time: %f seconds\n", totalTime);
        printf("Number of particles %d \nNumber of porcesses: %d\n", num_particles, numtasks);

        /* Write the output to a file for later correctness evaluation by comparing with sequential output */
        FILE *fileWrite = fopen("parallel_output_ring.txt", "w");
        if (fileWrite != NULL) {
            fwrite(all_particles, sizeof(Particle) * num_particles, 1, fileWrite);
            fclose(fileWrite);
        }

        free(all_particles);
    }

    free(my_portion);
    free(buf1);
    free(buf2);
    free(Fx);
    free(Fy);
    free(Fz);
    free(dim_portions);
    free(displ);

    return 0;
}

/* Equal distribution of work among tasks */
void compute_equal_workload_for_each_task(int *dim_portions, int *displs, int arraysize, int numtasks) {
    for (int i = 0; i < numtasks; i++) {
        dim_portions[i] = (arraysize / numtasks) +
                          ((i < (arraysize % numtasks)) ? 1 : 0);
    }

    // Set the displacements array: each index represents the start_offset of a task
    int offset = 0;
    for (int i = 0; i < numtasks; i++) {
        displs[i] = offset;
        offset += dim_portions[i];
    }
}

/* Conversion from string to integer */
int convertStringToInt(char *str) {
    char *endptr;
    long val;
    errno = 0;  // To distinguish success/failure after the call

    val = strtol(str, &endptr, 10);

    /* Check for possible errors */
    if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) {
        perror("strtol");
        exit(EXIT_FAILURE);
    }

    if (endptr == str) {
        fprintf(stderr, "No digits were found\n");
        exit(EXIT_FAILURE);
    }

    /* If we are here, strtol() has converted a number successfully */
    return (int)val;
}
