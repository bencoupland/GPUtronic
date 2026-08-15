    
    memset(g_throttle, 0, sizeof(ThrottleControl));
    g_throttle->q_fraction = 1.0;  
    g_throttle->running_flag = 1;
  

    printf("[GPUTronic] Starting control loop thread... (Q-axis only, pure work counter feedback)\n");
    
    pthread_t control_thread;
    if (pthread_create(&control_thread, NULL, control_loop_thread, NULL) != 0) {
        fprintf(stderr, "[ERROR] Failed to create control thread\n");
        return 1;
    }
