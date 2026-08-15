#include <stdio.h>
#include <string.h>

// Define minimal nvml structures based on driver API
typedef enum {
    NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

typedef struct {
    unsigned int temp;        // Temperature in Celsius
    unsigned int fanSpeedPct; // Fan speed percentage (optional)
} nvmlTemperature_t;

int main() {
    printf("nvmlTemperature_t struct:\n");
    printf("  temp field at offset: %zu bytes\n", sizeof(((nvmlTemperature_t*)0)->temp));
    return 0;
}
