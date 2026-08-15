#!/usr/bin/env python3
"""
GPUTronic Dyno Sweep Analysis Script
Calculates R² value and linear regression for thread fraction vs throughput

Usage: python3 analyze_dyno.py < dyno_sweep_output.txt
"""

import sys
import math

def parse_dyno_data(lines):
    """Parse dyno sweep output to extract q_fraction and throughput data."""
    data = []
    
    for line in lines:
        line = line.strip()
        
        # Look for lines like "q=0.1     | 123456789   | 1234.56         | 95.2%"
        if line.startswith('q='):
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    q_str = parts[0].strip()
                    throughput_str = parts[2].strip().split()[0]  # First number
                    
                    # Extract q value (e.g., "q=0.1" -> 0.1)
                    q = float(q_str.split('=')[1])
                    
                    # Extract throughput (e.g., "1234.56" -> 1234.56)
                    throughput = float(throughput_str)
                    
                    data.append((q, throughput))
                except (ValueError, IndexError):
                    continue
    
    return data

def linear_regression(x_vals, y_vals):
    """Calculate linear regression parameters."""
    n = len(x_vals)
    
    if n < 2:
        return None, None, None
    
    # Calculate means
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    
    # Calculate slope and intercept
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)
    
    if denominator == 0:
        return None, None, None
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calculate R² (coefficient of determination)
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_vals, y_vals))
    
    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared

def main():
    print("\n" + "="*70)
    print("  GPUTronic Dyno Sweep Analysis")
    print("="*70 + "\n")
    
    # Read input from stdin
    lines = sys.stdin.readlines()
    
    # Parse data
    data = parse_dyno_data(lines)
    
    if not data:
        print("[ERROR] No valid dyno sweep data found in input")
        print("Expected format: q=0.1     | 123456789   | 1234.56         | 95.2%")
        sys.exit(1)
    
    print(f"[Dyno] Parsed {len(data)} data points\n")
    
    # Extract x and y values
    q_values = [d[0] for d in data]
    throughput_values = [d[1] for d in data]
    
    # Perform linear regression
    slope, intercept, r_squared = linear_regression(q_values, throughput_values)
    
    if slope is None:
        print("[ERROR] Could not perform linear regression (insufficient data)")
        sys.exit(1)
    
    # Print results
    print("-"*70)
    print("  LINEAR REGRESSION RESULTS")
    print("-"*70 + "\n")
    
    print(f"  Slope:     {slope:.4f} (throughput units per q unit)")
    print(f"  Intercept: {intercept:.4f} (baseline throughput)")
    print(f"  R² value:  {r_squared:.6f}\n")
    
    # Pass/fail assessment
    print("-"*70)
    print("  LINEARITY ASSESSMENT")
    print("-"*70 + "\n")
    
    if r_squared >= 0.95:
        status = "✓ PASS"
        quality = "Excellent linearity"
    elif r_squared >= 0.90:
        status = "~ ACCEPTABLE"
        quality = "Good linearity, minor non-linearities detected"
    else:
        status = "✗ FAIL"
        quality = "Poor linearity - actuator model may need adjustment"
    
    print(f"  R² ≥ 0.95: {status}")
    print(f"  Quality:   {quality}\n")
    
    # Print regression equation
    print("-"*70)
    print("  REGRESSION EQUATION")
    print("-"*70 + "\n")
    print(f"  throughput = {slope:.4f} × q + {intercept:.4f}\n")
    
    # Calculate predicted vs actual for each point
    print("-"*70)
    print("  DATA POINTS WITH PREDICTIONS")
    print("-"*70 + "\n")
    print(f"  {'q':>6} | {'Actual':>12} | {'Predicted':>12} | {'Error (%)':>10}")
    print("  " + "-"*50)
    
    max_error = 0.0
    for q, actual in data:
        predicted = slope * q + intercept
        error_pct = abs(actual - predicted) / actual * 100 if actual > 0 else 0
        max_error = max(max_error, error_pct)
        
        print(f"  {q:>6.2f} | {actual:>12.2f} | {predicted:>12.2f} | {error_pct:>9.2f}%")
    
    print("\n" + "-"*70)
    print("  SUMMARY")
    print("-"*70 + "\n")
    print(f"  Max deviation from linear: {max_error:.2f}%")
    print(f"  Linearity quality:         {quality}")
    print(f"  Overall assessment:        {status}\n")
    
    # Recommendations
    if r_squared < 0.95:
        print("-"*70)
        print("  RECOMMENDATIONS")
        print("-"*70 + "\n")
        
        if r_squared < 0.80:
            print("  ⚠ Thread fraction actuator model may be incorrect")
            print("    Consider:")
            print("    - Checking fmod pattern in kernel")
            print("    - Verifying thread distribution across SMs")
            print("    - Testing with different WORK_UNITS_PER_THREAD values\n")
        else:
            print("  ℹ Minor non-linearities detected (acceptable for PoC)")
            print("    Consider gain scheduling if production deployment:\n")
            print("      q < 0.3:  Use Kp=0.25, Ki=0.04")
            print("      q 0.3-0.7: Use Kp=0.30, Ki=0.05")
            print("      q > 0.7:  Use Kp=0.35, Ki=0.06\n")
    
    # Plot data (for manual plotting if needed)
    print("-"*70)
    print("  PLOT DATA (copy to gnuplot or matplotlib)")
    print("-"*70 + "\n")
    print("# q_fraction throughput predicted")
    for q, actual in data:
        predicted = slope * q + intercept
        print(f"{q:.2f} {actual:.2f} {predicted:.2f}")
    
    print("\n" + "="*70)
    print("  Analysis complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
