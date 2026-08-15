// Example: How a real workload would use GPUTronic v26
// Compile with: nvcc -I../include basic_integration.c ../src/gputronic_v26_integration.c -o demo

#include "gputronic.h"
#include <stdio.h>
#include <unistd.h>

int main() {
    GPUTronicHandle* gov = gputronic_init(0.35f, 0.04f, 1.0f, 84);
    gputronic_start(gov);

    printf("Running controlled workload for 5 seconds...\n");
    for (int i = 0; i < 50; i++) {
        usleep(100000);
        printf("Z=%.3f  sleep=%6d ns\n",
               gputronic_get_z(gov),
               gputronic_get_sleep_ns(gov));
    }

    gputronic_stop(gov);
    printf("GPUTronic integration demo complete.\n");
    return 0;
}
