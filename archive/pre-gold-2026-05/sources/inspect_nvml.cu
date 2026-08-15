// Quick inspection of NVML structures
#include <stdio.h>
#include <nvml.h>

void inspect_struct(const char* name) {
    printf("%s:\n", name);
}

int main() {
    nvmlInit();
    inspect_struct("Running nvmlInit OK");
    
    // The API docs say this function returns temperature in temp member
    // But the struct definition might be different...
    // Let's just use the return value as the error code instead
    printf("\nTrying alternative approach...\n");
    
    unsigned int temp_int = 0;
    nvmlReturn_t ret = nvmlDeviceGetTemperatureV(NULL, &temp_int);
    printf("nvmlDeviceGetTemperatureV with dummy: %d\n", ret);
    
    nvmlShutdown();
    return 0;
}
