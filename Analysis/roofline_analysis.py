#!/usr/bin/env python3
"""
Complete Roofline Analysis for Bellman-Ford Implementation
Saves organized outputs in Analysis folder with version subdirectories
Usage: python3 roofline_analysis.py [path_to_results.csv]
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os
import re
import shutil

# ============================================================
# BASE ANALYSIS DIRECTORY - Change this to your Analysis folder
# ============================================================
BASE_ANALYSIS_DIR = "/home/fatima/PDC_HYBRID/Analysis"
# ============================================================

# Hardware peaks from your LIKWID measurements
PEAK_FLOPS = 53.96  # GFLOPS (AVX2+FMA, 4 threads)
PEAK_BANDWIDTH = 21.3  # GB/s (DDR4-2667 theoretical)

def extract_version(program_name):
    """Extract version from program name (v1, v3, v6, v7, etc.)"""
    match = re.search(r'v\d+', program_name)
    if match:
        return match.group(0)
    return "unknown"

def calculate_performance(V, D, time_seconds):
    """Calculate GFLOPS, bandwidth, and arithmetic intensity from graph metrics"""
    E = V * V * D  # Total edges
    flops_per_edge = 3  # add + comparison + conditional
    total_flops = E * flops_per_edge
    gflops = total_flops / (time_seconds * 1e9) if time_seconds > 0 else 0
    
    bytes_per_edge = 20  # dist[u](4) + weight(4) + adjncy(4) + dist[v](4) + write(4)
    total_bytes = E * bytes_per_edge
    bandwidth_gbs = total_bytes / (time_seconds * 1e9) if time_seconds > 0 else 0
    
    intensity = gflops / bandwidth_gbs if bandwidth_gbs > 0 else 0
    
    return {
        'edges': E,
        'gflops': gflops,
        'bandwidth': bandwidth_gbs,
        'intensity': intensity,
        'time': time_seconds
    }

def parse_results_csv(csv_path):
    """Parse results.csv and extract Bellman-Ford measurements"""
    results = []
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found - {csv_path}")
        return results
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 11:
                continue
            
            try:
                program = row[3]
                V = int(row[4])
                D = float(row[5])
                bf_seconds = float(row[10])
                
                perf = calculate_performance(V, D, bf_seconds)
                results.append({
                    'program': program,
                    'version': extract_version(program),
                    'V': V,
                    'D': D,
                    'time': bf_seconds,
                    'edges': perf['edges'],
                    'gflops': perf['gflops'],
                    'bandwidth': perf['bandwidth'],
                    'intensity': perf['intensity']
                })
            except (ValueError, IndexError):
                continue
    
    return results

def save_text_output(results, version, output_dir):
    """Save text summary to file"""
    txt_file = os.path.join(output_dir, f"{version}_roofline_summary.txt")
    
    with open(txt_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write(f"ROOFLINE ANALYSIS - {version.upper()}\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("BELLMAN-FORD PERFORMANCE SUMMARY\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'V':<8} {'D':<8} {'Time(s)':<12} {'GFLOPS':<12} {'Intensity':<12} {'Bandwidth(GB/s)':<15}\n")
        f.write("-" * 90 + "\n")
        
        for r in results:
            f.write(f"{r['V']:<8} {r['D']:<8.2f} {r['time']:<12.4f} "
                   f"{r['gflops']:<12.4f} {r['intensity']:<12.4f} {r['bandwidth']:<15.2f}\n")
        
        f.write("\n" + "=" * 90 + "\n")
        f.write("HARDWARE LIMITS\n")
        f.write("=" * 90 + "\n")
        f.write(f"CPU: Intel Core i7-10610U @ 1.80GHz\n")
        f.write(f"Peak GFLOPS (AVX2+FMA):        {PEAK_FLOPS:.2f}\n")
        f.write(f"Peak Memory Bandwidth (GB/s):  {PEAK_BANDWIDTH:.2f}\n")
        f.write(f"Ridge Point (FLOP/byte):       {(PEAK_FLOPS/PEAK_BANDWIDTH):.2f}\n")
        
        if results:
            avg_gflops = sum(r['gflops'] for r in results) / len(results)
            avg_intensity = sum(r['intensity'] for r in results) / len(results)
            peak_possible = PEAK_BANDWIDTH * avg_intensity
            
            f.write("\n" + "=" * 90 + "\n")
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("=" * 90 + "\n")
            f.write(f"Average GFLOPS:              {avg_gflops:.4f}\n")
            f.write(f"Average Arithmetic Intensity: {avg_intensity:.4f} FLOP/byte\n")
            f.write(f"Max possible @ this intensity: {peak_possible:.2f} GFLOPS\n")
            f.write(f"Efficiency vs bandwidth limit: {(avg_gflops/peak_possible)*100:.1f}%\n")
            f.write(f"Efficiency vs compute peak:    {(avg_gflops/PEAK_FLOPS)*100:.1f}%\n")
            
            if avg_intensity < (PEAK_FLOPS/PEAK_BANDWIDTH):
                f.write("\n✓ CONCLUSION: This implementation is MEMORY-BOUND\n")
                f.write("  Performance is limited by DRAM bandwidth, not CPU compute\n")
                f.write("  Improvement should focus on reducing memory traffic\n")
            else:
                f.write("\n✓ CONCLUSION: This implementation is COMPUTE-BOUND\n")
                f.write("  Performance is limited by CPU compute capacity\n")
    
    print(f"  ✓ Text output saved: {txt_file}")
    return txt_file

def create_roofline_plot(results, version, output_dir):
    """Create roofline plot for specific version"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # X-axis: Arithmetic Intensity (log scale)
    x = np.logspace(-2, 2, 1000)
    
    # Roofline lines
    bandwidth_line = PEAK_BANDWIDTH * x
    compute_ceiling = np.full_like(x, PEAK_FLOPS)
    
    ax.loglog(x, bandwidth_line, 'b-', lw=2.5, 
              label=f'Bandwidth Ceiling ({PEAK_BANDWIDTH} GB/s)')
    ax.loglog(x, compute_ceiling, 'r-', lw=2.5, 
              label=f'Compute Ceiling ({PEAK_FLOPS} GFLOPS)')
    
    # Ridge point
    ridge = PEAK_FLOPS / PEAK_BANDWIDTH
    ax.axvline(x=ridge, color='g', linestyle='--', lw=1.5, alpha=0.8,
               label=f'Ridge Point ({ridge:.2f} FLOP/byte)')
    
    # Plot points for different graph sizes
    sizes = sorted(set(r['V'] for r in results))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    for i, size in enumerate(sizes):
        size_results = [r for r in results if r['V'] == size]
        for r in size_results:
            ax.plot(r['intensity'], r['gflops'], 'o', 
                   markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                   color=colors[i], label=f'V={size}' if r == size_results[0] else "")
    
    # Fill memory-bound region
    ax.axvspan(0.001, ridge, alpha=0.15, color='yellow', label='Memory-Bound Region')
    ax.axvspan(ridge, 100, alpha=0.15, color='lightblue', label='Compute-Bound Region')
    
    # Theoretical Bellman-Ford region
    theo_low, theo_high = 0.10, 0.19
    ax.axvspan(theo_low, theo_high, alpha=0.2, color='gray', 
               label=f'Typical BF Range ({theo_low}-{theo_high} FLOP/byte)')
    
    # Labels and title
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=14)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax.set_title(f'Roofline Model - {version.upper()} Implementation\nIntel Core i7-10610U', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='lower right', fontsize=10)
    
    ax.set_xlim(0.02, 50)
    ax.set_ylim(0.1, 100)
    
    plt.tight_layout()
    
    # Save plot
    png_file = os.path.join(output_dir, f"{version}_roofline_plot.png")
    plt.savefig(png_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Roofline plot saved: {png_file}")
    plt.close(fig)
    
    return png_file

def create_all_versions_plot(all_results_by_version, output_dir):
    """Create a combined roofline plot with all versions"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # X-axis: Arithmetic Intensity (log scale)
    x = np.logspace(-2, 2, 1000)
    
    # Roofline lines
    bandwidth_line = PEAK_BANDWIDTH * x
    compute_ceiling = np.full_like(x, PEAK_FLOPS)
    
    ax.loglog(x, bandwidth_line, 'b-', lw=2.5, alpha=0.7,
              label=f'Bandwidth Ceiling ({PEAK_BANDWIDTH} GB/s)')
    ax.loglog(x, compute_ceiling, 'r-', lw=2.5, alpha=0.7,
              label=f'Compute Ceiling ({PEAK_FLOPS} GFLOPS)')
    
    # Ridge point
    ridge = PEAK_FLOPS / PEAK_BANDWIDTH
    ax.axvline(x=ridge, color='g', linestyle='--', lw=1.5, alpha=0.8,
               label=f'Ridge Point ({ridge:.2f} FLOP/byte)')
    
    # Color map for versions
    versions = sorted(all_results_by_version.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(versions)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, version in enumerate(versions):
        results = all_results_by_version[version]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        for r in results:
            ax.plot(r['intensity'], r['gflops'], marker, 
                   markersize=10, markeredgecolor='black', markeredgewidth=1,
                   color=color, label=f"{version.upper()} (V={r['V']})")
    
    # Fill memory-bound region
    ax.axvspan(0.001, ridge, alpha=0.1, color='yellow', label='Memory-Bound Region')
    ax.axvspan(ridge, 100, alpha=0.1, color='lightblue', label='Compute-Bound Region')
    
    # Labels
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=14)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax.set_title('Roofline Model - All Bellman-Ford Implementations\nIntel Core i7-10610U', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # Legend - handle too many labels
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 15:
        ax.legend(handles[:15], labels[:15], loc='lower right', fontsize=9)
    else:
        ax.legend(loc='lower right', fontsize=10)
    
    ax.set_xlim(0.02, 50)
    ax.set_ylim(0.1, 100)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_png = os.path.join(output_dir, "all_versions_roofline_plot.png")
    plt.savefig(combined_png, dpi=200, bbox_inches='tight')
    print(f"\n✓ Combined plot saved: {combined_png}")
    plt.close(fig)
    
    return combined_png

def main():
    # Create base analysis directory if it doesn't exist
    os.makedirs(BASE_ANALYSIS_DIR, exist_ok=True)
    
    # Get CSV path from command line or use default
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # If no argument provided, show usage
        print("\n" + "=" * 60)
        print("USAGE: python3 roofline_analysis.py <path_to_results.csv>")
        print("=" * 60)
        print("\nExamples:")
        print("  python3 roofline_analysis.py /home/fatima/PDC_HYBRID/V1_OMP/data/results.csv")
        print("  python3 roofline_analysis.py /home/fatima/PDC_HYBRID/V3_SIMD/data/results.csv")
        print("  python3 roofline_analysis.py /home/fatima/PDC_HYBRID/V6_MP+MPI/data/results_v6.csv")
        return
    
    print(f"\n📂 Reading results from: {csv_path}")
    
    # Parse results
    all_results = parse_results_csv(csv_path)
    
    if not all_results:
        print("\n❌ NO DATA FOUND! Please check your CSV path and format.")
        print("   Expected columns: program, vertices_num, density, bf_seconds")
        return
    
    # Get version from first result
    version = all_results[0]['version']
    
    # Create version-specific subdirectory in Analysis folder
    version_dir = os.path.join(BASE_ANALYSIS_DIR, f"{version}_analysis")
    os.makedirs(version_dir, exist_ok=True)
    
    print(f"📁 Output directory: {version_dir}")
    print(f"📊 Found {len(all_results)} measurements for {version.upper()}")
    
    # Process this version
    print("\n" + "=" * 60)
    print(f"PROCESSING {version.upper()}")
    print("=" * 60)
    
    # Save text output
    save_text_output(all_results, version, version_dir)
    
    # Create roofline plot
    create_roofline_plot(all_results, version, version_dir)
    
    # Save CSV of results for this version
    csv_file = os.path.join(version_dir, f"{version}_performance_data.csv")
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Version', 'V', 'D', 'Time(s)', 'GFLOPS', 'Intensity', 'Bandwidth(GB/s)'])
        for r in all_results:
            writer.writerow([r['version'], r['V'], r['D'], f"{r['time']:.4f}", 
                           f"{r['gflops']:.6f}", f"{r['intensity']:.6f}", 
                           f"{r['bandwidth']:.2f}"])
    print(f"  ✓ CSV data saved: {csv_file}")
    
    print("\n" + "=" * 60)
    print(f" ANALYSIS COMPLETE FOR {version.upper()}!")
    print("=" * 60)
    print(f"\n All outputs saved to: {version_dir}")
    print("\nGenerated files:")
    print(f"  - {version}_roofline_summary.txt")
    print(f"  - {version}_roofline_plot.png")
    print(f"  - {version}_performance_data.csv")

if __name__ == "__main__":
    main()