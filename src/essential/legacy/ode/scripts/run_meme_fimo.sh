#!/bin/bash
#
# Script to run MEME and FIMO using the official MEME Suite Docker container
# Usage: ./run_meme_fimo.sh <input_fasta> <output_dir> <genome_fasta> <bfile> [options]
#

set -e  # Exit on error

# Configuration
MEME_IMAGE="memesuite/memesuite:latest"
BEDTOOLS_IMAGE="biocontainers/bedtools:2.25.0"
SAMTOOLS_IMAGE="biocontainers/samtools:v1.7.0_cv4"
HOST_MOUNT_DIR="/ewsc/pboyeau" # Common parent directory for data

# Parse arguments
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <input_fasta> <output_dir> <genome_fasta> <bfile> <gff_file> [meme_options]"
    echo ""
    echo "Example:"
    echo "  $0 Cra/ri_sites.fasta Cra/output ecoli_genome.fasta ../U00096.3.bfile genes.gff \\"
    echo "     '-mod oops -nmotifs 1 -minw 12 -maxw 18 -maxsize 100000 -norand -seed 10'"
    echo ""
    echo "This script will:"
    echo "  1. Pull the MEME Suite Docker image (if needed)"
    echo "  2. Run MEME to discover motifs from input sequences"
    echo "  3. Run FIMO to scan the genome with discovered motifs"
    echo "  4. Map binding sites to genes using bedtools"
    exit 1
fi

INPUT_FASTA="$1"
OUTPUT_DIR="$2"
GENOME_FASTA="$3"
BFILE="$4"
GFF_FILE="$5"
OTHER_MEME_OPTIONS="${6:-"-mod oops -nmotifs 1 -minw 12 -maxw 18 -maxsize 100000 -norand -seed 10"}"
MEME_OPTIONS="-bfile $BFILE $OTHER_MEME_OPTIONS"

# Validate inputs
if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: Input FASTA file not found: $INPUT_FASTA"
    exit 1
fi

if [ ! -f "$GENOME_FASTA" ]; then
    echo "Error: Genome FASTA file not found: $GENOME_FASTA"
    exit 1
fi

if [ ! -f "$BFILE" ]; then
    echo "Error: Background file not found: $BFILE"
    exit 1
fi

if [ ! -f "$GFF_FILE" ]; then
    echo "Error: GFF annotation file not found: $GFF_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "MEME Suite Analysis Pipeline"
echo "============================================"
echo "Input FASTA:    $INPUT_FASTA"
echo "Output Dir:     $OUTPUT_DIR"
echo "Genome FASTA:   $GENOME_FASTA"
echo "GFF File:       $GFF_FILE"
echo "MEME Options:   $MEME_OPTIONS"
echo "============================================"
echo ""

# Step 1: Pull Docker images if needed
# echo "[1/7] Pulling Docker images..."
# echo "  - MEME Suite..."
# docker pull "$MEME_IMAGE"
# echo "  - bedtools..."
# docker pull "$BEDTOOLS_IMAGE"
# echo "  - samtools..."
# docker pull "$SAMTOOLS_IMAGE"
# echo ""

# Remove existing meme_output directory to prevent errors
if [ -d "$OUTPUT_DIR/meme_output" ]; then
    echo "Removing existing meme_output directory..."
    rm -rf "$OUTPUT_DIR/meme_output"
fi

# Step 2: Run MEME to discover motifs
echo "[2/7] Running MEME to discover motifs..."
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$MEME_IMAGE" \
    meme "$INPUT_FASTA" \
    -oc "$OUTPUT_DIR/meme_output" \
    -dna \
    $MEME_OPTIONS

if [ ! -f "$OUTPUT_DIR/meme_output/meme.txt" ]; then
    echo "Error: MEME did not produce output file"
    exit 1
fi

echo "MEME completed successfully!"
echo "Motif file: $OUTPUT_DIR/meme_output/meme.txt"
echo ""

# Step 3: Run FIMO to find binding sites in genome
echo "[3/7] Running FIMO to scan genome..."
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$MEME_IMAGE" \
    fimo \
    --oc "$OUTPUT_DIR/fimo_output" \
    --thresh 1e-4 \
    "$OUTPUT_DIR/meme_output/meme.txt" \
    "$GENOME_FASTA"

if [ ! -f "$OUTPUT_DIR/fimo_output/fimo.tsv" ]; then
    echo "Error: FIMO did not produce output file"
    exit 1
fi

echo "FIMO completed successfully!"
echo "Results file: $OUTPUT_DIR/fimo_output/fimo.tsv"
echo ""

# Step 4: Convert FIMO output to BED format
echo "[4/7] Converting FIMO output to BED format..."
BED_FILE="$OUTPUT_DIR/fimo_output/fimo.bed"
BED_FILE_UNSORTED="$OUTPUT_DIR/fimo_output/fimo_unsorted.bed"

# Convert TSV to BED
tail -n +2 "$OUTPUT_DIR/fimo_output/fimo.tsv" | \
    grep -v '^#' | \
    grep -v '^$' | \
    awk 'BEGIN{OFS="\t"} {print $3, $4-1, $5, $1"|"$10, $8, $6}' > "$BED_FILE_UNSORTED"

if [ ! -f "$BED_FILE_UNSORTED" ]; then
    echo "Error: Failed to create BED file"
    exit 1
fi

# Sort BED file
echo "Sorting BED file..."
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$BEDTOOLS_IMAGE" \
    bedtools sort -i "$BED_FILE_UNSORTED" > "$BED_FILE"

if [ ! -f "$BED_FILE" ]; then
    echo "Error: Failed to sort BED file"
    exit 1
fi

echo "BED file created and sorted: $BED_FILE"
echo ""

# Step 5: Create genome index
echo "[5/7] Creating genome index..."
GENOME_SIZES="$OUTPUT_DIR/genome.sizes"
# if [ ! -f "${GENOME_FASTA}.fai" ]; then
echo "Indexing genome with samtools..."
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$SAMTOOLS_IMAGE" \
    samtools faidx "$GENOME_FASTA"
# fi

if [ -f "${GENOME_FASTA}.fai" ]; then
    awk 'BEGIN{OFS="\t"} {print $1, $2}' "${GENOME_FASTA}.fai" > "$GENOME_SIZES"
    echo "Genome sizes file created"
else
    echo "Error: Failed to create genome index"
    exit 1
fi

# Sort GFF file for bedtools
SORTED_GFF="$OUTPUT_DIR/genes_sorted.gff"
echo "Sorting GFF file for bedtools..."
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$BEDTOOLS_IMAGE" \
    bedtools sort -i "$GFF_FILE" > "$SORTED_GFF"

if [ ! -f "$SORTED_GFF" ]; then
    echo "Error: Failed to sort GFF file"
    exit 1
fi

echo "GFF file sorted"
echo ""

# Step 6: Map binding sites to genes
echo "[6/7] Mapping binding sites to genes..."

# Create promoter regions (500bp upstream of genes)
PROMOTERS_FILE="$OUTPUT_DIR/promoters.bed"
echo "Creating promoter regions (500bp upstream)..."
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$BEDTOOLS_IMAGE" \
    bedtools flank -i "$SORTED_GFF" -g "$GENOME_SIZES" -l 500 -r 0 -s > "$PROMOTERS_FILE"

if [ ! -f "$PROMOTERS_FILE" ] || [ ! -s "$PROMOTERS_FILE" ]; then
    echo "Warning: Could not create promoter regions"
fi

# Find sites overlapping genes
GENES_OUTPUT="$OUTPUT_DIR/sites_in_genes.tsv"
echo "Finding sites in gene coding regions..."
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$BEDTOOLS_IMAGE" \
    bedtools intersect -a "$BED_FILE" -b "$SORTED_GFF" -wa -wb > "$GENES_OUTPUT"

GENE_COUNT=$(wc -l < "$GENES_OUTPUT")
echo "Found $GENE_COUNT binding sites overlapping genes"

# Find sites in promoter regions
if [ -f "$PROMOTERS_FILE" ] && [ -s "$PROMOTERS_FILE" ]; then
    PROMOTERS_OUTPUT="$OUTPUT_DIR/sites_in_promoters.tsv"
    echo "Finding sites in promoter regions..."
    docker run --rm \
        --userns=keep-id \
        --user "$(id -u):$(id -g)" \
        -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
        "$BEDTOOLS_IMAGE" \
        bedtools intersect -a "$BED_FILE" -b "$PROMOTERS_FILE" -wa -wb > "$PROMOTERS_OUTPUT"
    
    PROMOTER_COUNT=$(wc -l < "$PROMOTERS_OUTPUT")
    echo "Found $PROMOTER_COUNT binding sites in promoter regions"
fi
echo ""

# Step 7: Find closest gene to each binding site
echo "[7/7] Finding closest gene for each binding site..."
CLOSEST_OUTPUT="$OUTPUT_DIR/sites_closest_genes.tsv"
docker run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    -v "$HOST_MOUNT_DIR:$HOST_MOUNT_DIR" \
    "$BEDTOOLS_IMAGE" \
    bedtools closest -a "$BED_FILE" -b "$SORTED_GFF" -d > "$CLOSEST_OUTPUT"

if [ ! -f "$CLOSEST_OUTPUT" ]; then
    echo "Error: Failed to find closest genes"
    exit 1
fi

echo "Closest gene analysis complete"
echo ""

# Summary
echo "============================================"
echo "Analysis Complete!"
echo "============================================"
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Key output files:"
echo "  - Motifs:              $OUTPUT_DIR/meme_output/meme.txt"
echo "  - Motifs (HTML):       $OUTPUT_DIR/meme_output/meme.html"
echo "  - Binding sites:       $OUTPUT_DIR/fimo_output/fimo.tsv"
echo "  - Sites (HTML):        $OUTPUT_DIR/fimo_output/fimo.html"
echo "  - Sites (BED):         $OUTPUT_DIR/fimo_output/fimo.bed"
echo "  - Sites in genes:      $OUTPUT_DIR/sites_in_genes.tsv"
echo "  - Sites in promoters:  $OUTPUT_DIR/sites_in_promoters.tsv"
echo "  - Sites closest genes: $OUTPUT_DIR/sites_closest_genes.tsv"
echo "============================================"

