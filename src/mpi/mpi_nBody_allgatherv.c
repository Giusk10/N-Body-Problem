#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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
void bodyForce(Particle *all_particles, int startOffsetPortion, float dt, int dim_portion, int num_particles);
int convertStringToInt(char *str);

int main(int argc, char* argv[]) {
    MPI_Datatype particle_type;             // MPI datatype to communicate the "Particle" data type
    int numtasks;                           // Number of used processors
    int myrank;                             // Rank of the current process
    double start, end, iterStart, iterEnd;  // Variables used for measuring the total execution time and each iteration

    int *dim_portions;                      // Size of the workload portion for each process
    int *displ;                             // Starting offset of the workload portion for each process

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

    /* All processes maintain the full particles array — updated via MPI_Allgatherv */
    Particle *particles = (Particle*)malloc(num_particles * sizeof(Particle));

    for (int iteration = 0; iteration < I; iteration++) {

        MPI_Barrier(MPI_COMM_WORLD);  // Synchronize processes before starting to measure the iteration execution time
        iterStart = MPI_Wtime();

        if (iteration == 0) {
            // First iteration: all processors read the initial state of particles from a file
            FILE *fileRead = fopen("particles.txt", "r");
            if (fileRead == NULL) {
                printf("\nUnable to open the file.\n");
                exit(EXIT_FAILURE);
            }

            int particlesRead = fread(particles, sizeof(Particle) * num_particles, 1, fileRead);
            if (particlesRead == 0) {
                printf("ERROR: The number of particles to read is greater than the number of particles in the file\n");
                exit(EXIT_FAILURE);
            }

            fclose(fileRead);
        }
        /* For iteration > 0: all processes already have the up-to-date full particles array
           from the MPI_Allgatherv call at the end of the previous iteration — no Bcast needed. */

        bodyForce(particles, displ[myrank], dt, dim_portions[myrank], num_particles);

        /* Each process shares its updated portion with all others.
           MPI_IN_PLACE: each process reads its send data from its own position in the recvbuf. */
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       particles, dim_portions, displ, particle_type,
                       MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        iterEnd = MPI_Wtime();
        if (myrank == MASTER) printf("Iteration %d of %d completed in %f seconds\n", iteration + 1, I, (iterEnd - iterStart));
    }

    MPI_Barrier(MPI_COMM_WORLD);     // All processes have finished
    end = MPI_Wtime();               // Record the end time of execution
    MPI_Type_free(&particle_type);
    MPI_Finalize();

    if (myrank == MASTER) {
        double totalTime = end - start;
        double avgTime = totalTime / (double)(I);
        printf("\nAvg iteration time: %f seconds\n", avgTime);
        printf("Total time: %f seconds\n", totalTime);
        printf("Number of particles %d \nNumber of porcesses: %d\n", num_particles, numtasks);

        /* Write the output to a file for later correctness evaluation by comparing with sequential output */
        FILE *fileWrite = fopen("parallel_output_allgatherv.txt", "w");
        if (fileWrite != NULL) {
            fwrite(particles, sizeof(Particle) * num_particles, 1, fileWrite);
            fclose(fileWrite);
        }
    }

    free(dim_portions);
    free(displ);
    free(particles);

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

/* Function that performs computation on a specific workload portion */
void bodyForce(Particle *all_particles, int startOffsetPortion, float dt, int dim_portion, int num_particles) {
    for (int i = 0; i < dim_portion; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < num_particles; j++) {
            float dx = all_particles[j].x - all_particles[startOffsetPortion + i].x;
            float dy = all_particles[j].y - all_particles[startOffsetPortion + i].y;
            float dz = all_particles[j].z - all_particles[startOffsetPortion + i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        all_particles[startOffsetPortion + i].vx += dt * Fx;
        all_particles[startOffsetPortion + i].vy += dt * Fy;
        all_particles[startOffsetPortion + i].vz += dt * Fz;
    }

    // Integrate the positions of my portion
    for (int i = 0; i < dim_portion; i++) {
        all_particles[startOffsetPortion + i].x += all_particles[startOffsetPortion + i].vx * dt;
        all_particles[startOffsetPortion + i].y += all_particles[startOffsetPortion + i].vy * dt;
        all_particles[startOffsetPortion + i].z += all_particles[startOffsetPortion + i].vz * dt;
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
