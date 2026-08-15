/* GPUTronic Gold — minimal embed example */
#include "gputronic.h"
#include <stdio.h>
#include <unistd.h>

int main(void) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_demo.csv";

    GPUTronicHandle* gov = gputronic_create(&cfg);
    if (!gov) return 1;
    if (gputronic_start(gov) != 0) {
        gputronic_destroy(gov);
        return 1;
    }

    printf("Gold demo: 10s closed-loop\n");
    for (int i = 0; i < 10; i++) {
        usleep(1000000);
        gputronic_print_status(gov, i + 1);
    }

    gputronic_stop(gov);
    gputronic_destroy(gov);
    return 0;
}
