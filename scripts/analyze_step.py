#!/usr/bin/env python3
"""
GPUTronic Step Response Analysis Script
Calculates overshoot %, settling time, rise time from step response data

Usage: python3 analyze_step.py < step_response_output.txt
"""

import sys
import math

def parse_step_data(lines):
    """Parse step response output to extract time, q_fraction, Z, throughput."""
    data = []
    
    for line in lines:
        line = line.strip()
        
        # Look for lines like "100.5 | 0.50      | 1.234       | 987.65"
        try:
            parts = line.split('|')
            if len(parts) >= 4:
                time_str = parts[0].strip()
                q_str = parts[1].strip()
                z_str = parts[2].strip()
                throughput_str = parts[3].strip().split()[0]
                
                # Parse values
                time_ms = float(time_str)
                q_fraction = float(q_str)
                z_value = float(z_str)
                throughput = float(throughput_str)
                
                data.append({
                    'time': time_ms,
                    'q': q_fraction,
                    'z': z_value,
                    'throughput': throughput
                })
        except (ValueError, IndexError):
            continue
    
    return data

def calculate_metrics(data):
    """Calculate step response metrics."""
    if len(data) < 2:
        return None
    
    # Find the step point (where q changes significantly)
    step_time = None
    initial_q = None
    final_q = None
    
    for i, d in enumerate(data):
        if i > 0:
            prev_q = data[i-1]['q']
            curr_q = d['q']
            
            # Detect step change (threshold: 0.2)
            if abs(curr_q - prev_q) > 0.2:
                step_time = d['time']
                initial_q = prev_q
                final_q = curr_q
                break
    
    if step_time is None:
        return None
    
    # Find steady-state values before and after step
    pre_step_data = [d for d in data if d['time'] < step_time - 50]  # 50ms buffer
    post_step_data = [d for d in data if d['time'] > step_time + 200]  # Wait for settling
    
    if not pre_step_data or not post_step_data:
        return None
    
    initial_throughput = sum(d['throughput'] for d in pre_step_data) / len(pre_step_data)
    final_throughput = sum(d['throughput'] for d in post_step_data) / len(post_step_data)
    
    # Find peak throughput after step
    post_step_throughputs = [d['throughput'] for d in data if d['time'] > step_time]
    peak_throughput = max(post_step_throughputs) if post_step_throughputs else final_throughput
    
    # Calculate overshoot
    expected_change = final_throughput - initial_throughput
    actual_peak_change = peak_throughput - initial_throughput
    
    if expected_change != 0:
        overshoot_pct = (actual_peak_change / expected_change - 1) * 100
    else:
        overshoot_pct = 0.0
    
    # Calculate rise time (10% to 90%)
    target_10pct = initial_throughput + 0.1 * expected_change
    target_90pct = initial_throughput + 0.9 * expected_change
    
    rise_start_time = None
    rise_end_time = None
    
    for d in data:
        if d['time'] > step_time:
            if rise_start_time is None and d['throughput'] >= target_10pct:
                rise_start_time = d['time']
            elif rise_start_time is not None and d['throughput'] >= target_90pct:
                rise_end_time = d['time']
                break
    
    rise_time = (rise_end_time - rise_start_time) if (rise_start_time and rise_end_time) else None
    
    # Calculate settling time (within 2% of final value)
    lower_bound = final_throughput * 0.98
    upper_bound = final_throughput * 1.02
    
    settling_time = None
    for d in data:
        if d['time'] > step_time:
            # Check if all subsequent points are within bounds
            all_within_bounds = True
            for later_d in data[data.index(d):]:
                if not (lower_bound <= later_d['throughput'] <= upper_bound):
                    all_within_bounds = False
                    break
            
            if all_within_bounds:
                settling_time = d['time'] - step_time
                break
    
    return {
        'step_time': step_time,
        'initial_q': initial_q,
        'final_q': final_q,
        'initial_throughput': initial_throughput,
        'final_throughput': final_throughput,
        'peak_throughput': peak_throughput,
        'overshoot_pct': overshoot_pct,
        'rise_time': rise_time,
        'settling_time': settling_time
    }

def main():
    print("\n" + "="*70)
    print("  GPUTronic Step Response Analysis")
    print("="*70 + "\n")
    
    # Read input from stdin
    lines = sys.stdin.readlines()
    
    # Parse data
    data = parse_step_data(lines)
    
    if not data:
        print("[ERROR] No valid step response data found in input")
        print("Expected format: Time (ms) | q_frac | Z | Throughput (M/s)")
        sys.exit(1)
    
    print(f"[Step] Parsed {len(data)} data points\n")
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    if metrics is None:
        print("[ERROR] Could not detect step change or calculate metrics")
        sys.exit(1)
    
    # Print results
    print("-"*70)
    print("  STEP RESPONSE METRICS")
    print("-"*70 + "\n")
    
    print(f"  Step detected at:      t = {metrics['step_time']:.1f} ms")
    print(f"  q change:              {metrics['initial_q']:.2f} → {metrics['final_q']:.2f}")
    print(f"  Throughput change:     {metrics['initial_throughput']:.2f} → {metrics['final_throughput']:.2f} M/s")
    print(f"  Peak throughput:       {metrics['peak_throughput']:.2f} M/s\n")
    
    # Overshoot assessment
    print("-"*70)
    print("  OVERSHOOT ANALYSIS")
    print("-"*70 + "\n")
    
    overshoot = metrics['overshoot_pct']
    
    if overshoot < 0:
        status = "✓ PASS"
        quality = "Under-damped (no overshoot)"
    elif overshoot <= 15:
        status = "✓ PASS"
        quality = "Well-damped response"
    elif overshoot <= 25:
        status = "~ ACCEPTABLE"
        quality = "Moderate overshoot, may need tuning"
    else:
        status = "✗ FAIL"
        quality = "High overshoot - reduce Kp or increase damping"
    
    print(f"  Overshoot:             {overshoot:.1f}%")
    print(f"  Target:                <15%")
    print(f"  Status:                {status}")
    print(f"  Quality:               {quality}\n")
    
    # Rise time assessment
    print("-"*70)
    print("  RISE TIME ANALYSIS")
    print("-"*70 + "\n")
    
    if metrics['rise_time']:
        rise_time = metrics['rise_time']
        
        if rise_time < 30:
            status = "✓ FAST"
            quality = "Very responsive (may be too aggressive)"
        elif rise_time < 80:
            status = "✓ GOOD"
            quality = "Optimal bandwidth"
        else:
            status = "~ SLOW"
            quality = "May need higher Kp for faster response"
        
        print(f"  Rise time (10%→90%):   {rise_time:.1f} ms")
        print(f"  Target:                20-80 ms")
        print(f"  Status:                {status}")
        print(f"  Quality:               {quality}\n")
    else:
        print("  Rise time:             Could not calculate\n")
    
    # Settling time assessment
    print("-"*70)
    print("  SETTLING TIME ANALYSIS")
    print("-"*70 + "\n")
    
    if metrics['settling_time']:
        settling_time = metrics['settling_time']
        
        if settling_time < 50:
            status = "✓ EXCELLENT"
            quality = "Very fast recovery"
        elif settling_time < 100:
            status = "✓ GOOD"
            quality = "Fast enough for real-time control"
        elif settling_time < 200:
            status = "~ ACCEPTABLE"
            quality = "May need faster response for some workloads"
        else:
            status = "✗ SLOW"
            quality = "Too slow - increase Ki or reduce deadband"
        
        print(f"  Settling time (2%):    {settling_time:.1f} ms")
        print(f"  Target:                <100 ms")
        print(f"  Status:                {status}")
        print(f"  Quality:               {quality}\n")
    else:
        print("  Settling time:         Could not calculate (may not have settled)\n")
    
    # Stability assessment
    print("-"*70)
    print("  STABILITY ASSESSMENT")
    print("-"*70 + "\n")
    
    # Count how many metrics passed
    pass_count = 0
    
    if overshoot <= 25:
        pass_count += 1
    if metrics['rise_time'] and metrics['rise_time'] < 150:
        pass_count += 1
    if metrics['settling_time'] and metrics['settling_time'] < 300:
        pass_count += 1
    
    if pass_count == 3:
        overall = "✓ PASS - System is stable and well-tuned"
    elif pass_count >= 2:
        overall = "~ ACCEPTABLE - Minor tuning recommended"
    else:
        overall = "✗ FAIL - Significant retuning needed"
    
    print(f"  Metrics passed:        {pass_count}/3")
    print(f"  Overall assessment:    {overall}\n")
    
    # Recommendations
    if overshoot > 15 or (metrics['settling_time'] and metrics['settling_time'] > 200):
        print("-"*70)
        print("  RECOMMENDATIONS FOR TUNING")
        print("-"*70 + "\n")
        
        if overshoot > 30:
            print("  ⚠ High overshoot detected:")
            print("    - Reduce Kp from 0.3 to 0.2")
            print("    - Or add derivative term (PID instead of PI)\n")
        
        if metrics['settling_time'] and metrics['settling_time'] > 300:
            print("  ⚠ Slow settling time detected:")
            print("    - Increase Ki from 0.05 to 0.08")
            print("    - Or reduce Z-axis deadband width\n")
        
        if overshoot < 5 and metrics['rise_time'] and metrics['rise_time'] > 100:
            print("  ℹ System is over-damped:")
            print("    - Increase Kp from 0.3 to 0.4 for faster response\n")
    
    # Plot data (for manual plotting if needed)
    print("-"*70)
    print("  PLOT DATA (copy to gnuplot or matplotlib)")
    print("-"*70 + "\n")
    print("# Time(ms) | q_fraction | Z_value | Throughput(M/s)")
    for d in data:
        print(f"{d['time']:.1f} | {d['q']:.2f} | {d['z']:.3f} | {d['throughput']:.2f}")
    
    print("\n" + "="*70)
    print("  Analysis complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
