// Compile and run to inspect nvmlTemperature_v1_t structure
#include <stdio.h>
#include <nvml.h>

void print_struct_info() {
    printf("Testing nvmlDeviceGetTemperatureV API:\n");
    
    unsigned int temp;
    if (nvmlInit() == NVML_SUCCESS) {
        if (nvmlDeviceGetHandleByIndex(0, NULL) != NVML_SUCCESS) {
            printf("  No GPU found\n");
            nvmlShutdown();
            return;
        }
        
        // The API docs say this function takes a struct pointer
        // But maybe it returns the temp directly?
        printf("  Trying: unsigned int temp;\n");
        printf("         nvmlDeviceGetTemperatureV(device, &temp);\n");
        
        unsigned int temp_int;
        if (nvmlDeviceGetTemperatureV(NULL, &temp_int) != NVML_SUCCESS) {
            printf("  Call failed as expected with NULL device\n");
        } else {
            printf("  Call succeeded - temp_int = %u\n", temp_int);
        }
        
        // Or maybe it's nvmlDeviceGetTemperatureV2?
        printf("\n  Checking for nvmlDeviceGetTemperatureV2...\n");
    }
    
    nvmlShutdown();
}

int main() {
    print_struct_info();
    return 0;
}
