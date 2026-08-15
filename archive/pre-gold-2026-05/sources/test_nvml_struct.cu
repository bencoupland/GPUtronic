#include <nvml.h>
#include <stdio.h>

int main() {
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    
    // Try the new API
    printf("=== Testing nvmlDeviceGetTemperatureV ===\n");
    nvmlReturn_t ret = nvmlDeviceGetTemperatureV(device, NULL);
    printf("Return: %d\n", ret);
    
    // Try to find structure members by calling with different arguments
    unsigned int temp_int;
    ret = nvmlDeviceGetTemperatureV(device, &temp_int);
    printf("temp_int: %u, return: %d\n", temp_int, ret);
    
    nvmlShutdown();
    return 0;
}
