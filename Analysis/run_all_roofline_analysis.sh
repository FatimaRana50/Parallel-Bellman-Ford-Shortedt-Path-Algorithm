#!/bin/bash

# Base Analysis directory
ANALYSIS_DIR="/home/fatima/PDC_HYBRID/Analysis"
SCRIPT_DIR="$ANALYSIS_DIR"

# Make sure we're in the right directory
cd $ANALYSIS_DIR

echo "========================================="
echo "🚀 Running Roofline Analysis for ALL versions"
echo "========================================="

# List of CSV files for each version
declare -A versions=(
    ["V0part1"]="/home/fatima/PDC_HYBRID/parallel_bellman_ford/data/comparison_results.csv"
    
    ["v1"]="/home/fatima/PDC_HYBRID/V1_OMP/data/results.csv"
    ["v2"]="/home/fatima/PDC_HYBRID/V2_SIMD/data/results.csv"
    ["v3"]="/home/fatima/PDC_HYBRID/V3_OMP+SIMD/data/results.csv"
    ["v4"]="/home/fatima/PDC_HYBRID/V4_MPI/results.csv"
    ["v5"]="/home/fatima/PDC_HYBRID/V5_SIMD+MPI/results.csv"
    ["v6"]="/home/fatima/PDC_HYBRID/V6_MP+MPI/results.csv"
    ["v7"]="/home/fatima/PDC_HYBRID/V7_MP+MPI+SIMD/data/results.csv"
)

for version in "${!versions[@]}"; do
    csv_file="${versions[$version]}"
    if [ -f "$csv_file" ]; then
        echo ""
        echo "========================================="
        echo "📊 Processing $version from: $csv_file"
        echo "========================================="
        python3 "$SCRIPT_DIR/roofline_analysis.py" "$csv_file"
    else
        echo ""
        echo "⚠️  File not found for $version: $csv_file"
    fi
done

echo ""
echo "========================================="
echo "✅ ALL ANALYSES COMPLETE!"
echo "========================================="
echo ""
echo "📁 Output directories created in: $ANALYSIS_DIR"
echo ""
echo "Contents:"
ls -la $ANALYSIS_DIR/ | grep "_analysis"

EOF

chmod +x ~/PDC_HYBRID/Analysis/run_all_roofline_analysis.sh
