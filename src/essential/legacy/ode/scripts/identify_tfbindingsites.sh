#!/bin/bash
./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/Cra/ri_sites.fasta" \
  "/ewsc/pboyeau/results/Cra/ri_sites_meme_fimo" \
  "/ewsc/pboyeau/data/genomes/ASM584v2/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna" \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/U00096.3.bfile" \
  "/ewsc/pboyeau/data/genomes/ASM584v2/ncbi_dataset/data/GCF_000005845.2/genomic.gff"

#!/bin/bash

# GENOME_FASTA="/ewsc/pboyeau/data/genomes/ASM584v2/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna"
# BFILE="/ewsc/pboyeau/data/RegulonDB_PSSM/U00096.3.bfile"
# GFF_FILE="/ewsc/pboyeau/data/genomes/ASM584v2/ncbi_dataset/data/GCF_000005845.2/genomic.gff"

GENOME_FASTA="/ewsc/pboyeau/data/genomes/NC_000913.3/E_coli_K12_MG1655_NC_000913.3.fa"
BFILE="/ewsc/pboyeau/data/RegulonDB_PSSM/U00096.3.bfile"
GFF_FILE="/ewsc/pboyeau/data/genomes/NC_000913.3/E_coli_K12_MG1655_NC_000913.3.gff3"

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/Cra/ri_sites.fasta" \
  "/ewsc/pboyeau/results/Cra/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/IHF/ri_sites.fasta" \
  "/ewsc/pboyeau/results/IHF/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/RhaS/ri_sites.fasta" \
  "/ewsc/pboyeau/results/RhaS/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/TorR/ri_sites.fasta" \
  "/ewsc/pboyeau/results/TorR/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/XylR/ri_sites.fasta" \
  "/ewsc/pboyeau/results/XylR/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/PurR/ri_sites.fasta" \
  "/ewsc/pboyeau/results/PurR/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/SdiA/ri_sites.fasta" \
  "/ewsc/pboyeau/results/SdiA/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/CysB/ri_sites.fasta" \
  "/ewsc/pboyeau/results/CysB/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE


./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/CysB/ri_sites.fasta" \
  "/ewsc/pboyeau/results/CysB/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE


./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/RstA/ri_sites.fasta" \
  "/ewsc/pboyeau/results/RstA/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE



./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/BaeR/ri_sites.fasta" \
  "/ewsc/pboyeau/results/BaeR/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE


./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/RbsR/ri_sites.fasta" \
  "/ewsc/pboyeau/results/RbsR/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE


./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/Rob/ri_sites.fasta" \
  "/ewsc/pboyeau/results/Rob/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE


./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/Lrp/ri_sites.fasta" \
  "/ewsc/pboyeau/results/Lrp/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE

./run_meme_fimo.sh \
  "/ewsc/pboyeau/data/RegulonDB_PSSM/RcsB/ri_sites.fasta" \
  "/ewsc/pboyeau/results/RcsB/ri_sites_meme_fimo" \
  $GENOME_FASTA \
  $BFILE \
  $GFF_FILE


