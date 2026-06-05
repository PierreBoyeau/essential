# Rapp 2026 — E. coli CRISPRi Metabolomics Dataset

Source files live in `/workspace/data/Rapp_2026/`.

## Overview

Genome-scale CRISPRi knockdown screen in *E. coli* (iML1515 / 1515 genes).
Each targeted gene was knocked down with one sgRNA guide; OD600 growth curves
and untargeted LC-MS metabolomics were collected per knockdown (2–3 replicates).
Metabolite fold-changes are relative to wild-type under EZ Rich Medium.

---

## File descriptions

### mmc2.xlsx — Table S1: sgRNA library
**Sheet:** `Table_S1` | **Shape:** 1515 rows × 5 cols

One row per gene. Contains the guide sequence used for CRISPRi knockdown.

| Column | Description |
|---|---|
| Gene | Gene name |
| sgRNA Nr. | Guide identifier (e.g. `glyA #1`) |
| b-Nr. | Blattner number (b-number) |
| base pairing region | 20-nt protospacer sequence |
| Oligo sequence | Full cloning oligo including promoter, spacer, scaffold |

---

### mmc3.xlsx — Table S2: OD600 growth curves
**Sheet:** `Table_S2` | **Shape:** 4593 rows × 187 cols

Time-resolved OD600 measurements for every knockdown replicate.
Row 0 labels the time axis ("Time (hours)"); row 1 contains the actual column
headers and numeric time points; data starts at row 2.

| Column | Description |
|---|---|
| Replicate Nr. | 1, 2, or 3 |
| RXN Nr. | Reaction / well number |
| Gene | Targeted gene name |
| Guide Nr. | sgRNA guide identifier |
| b-Nr. | Blattner number |
| Plate ID | Plate identifier |
| 0 … 30.0 | OD600 at each time point (181 points, ~10 min spacing, 0–30 h) |

**Loading note:** use `load.py::load_growth_curves()` which handles the
two-row header and returns clean metadata + float OD columns.

---

### mmc4.xlsx — Table S3: Endpoint OD
**Sheet:** `Table_S3` | **Shape:** 3026 rows × 7 cols

Single endpoint OD600 measurement per replicate, recorded at harvest time
(when cells were pelleted for metabolomics extraction).

| Column | Description |
|---|---|
| Target gene | Gene name |
| b number | Blattner number |
| OD | OD600 at harvest |
| Plate ID | Plate identifier |
| Well | Well position |
| Replicate | R1 or R2 |
| Sample ID | Full sample identifier matching mmc5 column names |

---

### mmc5.xlsx — Table S4: Full metabolomics fold-change matrix
**Sheet:** `Table_S4` | **Shape:** 1880 rows × 3030 cols

Wide-format matrix: 1880 detected ions (m/z features) × 3026 samples.
Values are fold-changes relative to WT (same batch). Isobaric metabolites
share a row.

| Column | Description |
|---|---|
| Abbr | Metabolite abbreviation (iML1515 / custom; isobaric entries joined with `-`) |
| Metabolite | Full metabolite name |
| Mass | Measured m/z |
| Kegg | KEGG compound ID(s) |
| *Sample columns* | One column per sample, named `{gene}_{Rn}_{msAVxxx}_{Bn}` |

---

### mmc6.xlsx — Table S5: Significant metabolite hits (annotated)
**Sheet:** `TableS5` (+ `Legend`) | **Shape:** 1385 rows × 29 cols

Subset of metabolite-gene pairs that passed significance thresholds.
Includes biological annotation (FBA reactant status, pathway, operon effects).

Key columns: `Gene`, `Metabolite`, `Metabolite Abbreviation`, `Mean_FC`,
`R1_FC`, `R2_FC`, `Reactant`, `Subsystem`, `FBA`, `Substrate Abb`,
`Product Abb`, `Operon effect`, `Pathways`, `Position`,
`Upstream Accumulation`, `Downstream Accumulation`.

---

### mmc7.xlsx — Table S6: Significant hits with MS2 spectra and QC
**Sheet:** `Table_S6` (+ `Legend`) | **Shape:** 1256 rows × 54 cols

Extended version of S5 with full LC-MS/MS spectral evidence.

Key columns beyond S5: `QC passed`, `SMILES`, `ExpSpectrum`,
`retention time (sec)`, `fold-change`, `Deviation (Da)`,
`Scan # CE10/20/40`, `m/z MS2 fragments CE10/20/40`,
`norm. intensity MS2 fragments CE10/20/40`, `DataFile`.

---

### mmc8.xlsx — Table S7: Significant metabolite fold-changes per gene
**Sheet:** `Table_S7` | **Shape:** 9462 rows × 10 cols

Compact significant-hit table — one row per (gene, metabolite, ionisation mode).
Includes mean and per-replicate fold-changes and intensities.

| Column | Description |
|---|---|
| Gene | Targeted gene |
| Metabolite | Metabolite name + adduct in brackets |
| Mass | Precursor m/z |
| Mode | `pos` or `neg` (ionisation polarity) |
| Mean_FC | Mean fold-change (log2) |
| R1_FC / R2_FC | Per-replicate fold-changes |
| Mean_Int / R1_Int / R2_Int | Precursor ion intensities |

---

### mmc9.xlsx — Table S8: Non-annotated iML1515 m/z features
**Sheet:** `Table_S8` | **Shape:** 52 rows × 27 cols

SIRIUS/CSI:FingerID annotation results for features that could not be matched
to iML1515 but passed QC. Columns include `Polarity`, `ConfidenceScoreExact`,
`ConfidenceScoreApproximate`, `CSI:FingerIDScore`, `ZodiacScore`, `SiriusScore`.

---

### mmc10.xlsx — Table S9: Metabolite reference table
**Sheet:** `TableS9` | **Shape:** 802 rows × 6 cols

Cross-reference between abbreviations and database identifiers for all
annotated metabolites.

| Column | Description |
|---|---|
| Abbreviation | Short abbreviation used throughout the dataset |
| BIGG | BiGG database ID |
| Metabolite | Full name |
| KEGG | KEGG compound ID |
| Monoisotopic mass | Neutral monoisotopic mass (Da) |
| Neutral Formula | Molecular formula |

---

### mmc11.xlsx — Table S10: Gene → substrate metabolites
**Sheet:** `TableS10` | **Shape:** 7683 rows × 9 cols

Maps each targeted gene to the metabolites it acts on, derived from iML1515
reaction stoichiometry.

| Column | Description |
|---|---|
| gene | Gene name |
| Gene Number | b-number |
| Abbreviation | Metabolite abbreviation |
| Metabolite | Full name |
| Subsystem | Metabolic subsystem from iML1515 |
| KEGG | KEGG ID |
| BIGG | BiGG ID |
| Monoisotopic Mass | Neutral mass (Da) |
| Neutral Formula | Molecular formula |

---

### mmc12.xlsx — Table S11: Gene pathway positions
**Sheet:** `Table_S11` | **Shape:** 3340 rows × 11 cols

Position of each gene within its EcoCyc/BioCyc pathway.
Used to determine whether an accumulating metabolite is upstream or
downstream of the knocked-down enzyme.

| Column | Description |
|---|---|
| Position | Ordinal position in the pathway |
| Abb | Pathway abbreviation |
| Pathway | Full pathway name |
| GeneID | NCBI Gene ID |
| GeneAccession | b-number |
| GeneName | Gene name |
| ReactionId | BioCyc reaction ID |
| ReactionEC | EC number |
| EnzymaticActivity | Activity description |
| Evidence | Literature evidence code |
| Sub- or superpathway | Hierarchy flag |

---

### mmc13.xlsx — Table S12: Extended pathway-metabolite mapping
**Sheet:** `Table_S12` | **Shape:** 11851 rows × 18 cols

Extension of S11 that additionally lists the metabolites at each pathway
position (substrates/products), enabling upstream/downstream accumulation
analysis.

Additional columns over S11: `metAbb`, `metAbNames`, `subSys`, `KEGGID`,
`BIGG`, `mass`, `mass_13C`, `NeutralFormula`.

---

## FBA model

`iML1515.json` (in this directory) — E. coli iML1515 genome-scale metabolic
model (downloaded from BiGG). Helper functions in `/workspace/src/essential/fba.py`:

- `load_ecoli_rich_medium_model()` — loads the model with EZ Rich Medium
  constraints (all 20 amino acids + nucleobases + vitamins unlocked).
- `get_model_components_df()` — returns `(metabolites_df, reactions_df,
  genes_df)` DataFrames from the JSON.

---

## Dataset loaders (`load_rapp.py`)

All loaders read from `/workspace/data/Rapp_2026/`. Every function returns a
`pd.DataFrame` unless noted.

| Function | Source file | Returns | Shape |
|---|---|---|---|
| `load_sgrna_library()` | mmc2 | One row per gene: guide sequence + b-number | 1515 × 5 |
| `load_growth_curves()` | mmc3 | `(curves_df, time_h)` — metadata + OD columns named by float hour; `time_h` is ndarray of 181 time points | 4591 × 187 |
| `load_endpoint_od()` | mmc4 | Harvest-time OD600 per replicate; `Sample_ID` matches mmc5 column names | 3026 × 7 |
| `load_metabolomics_matrix()` | mmc5 | Full fold-change matrix: first 4 cols are metadata (`Abbrev`, `Metabolite`, `Mass`, `KEGG`), rest are samples | 1880 × 3030 |
| `load_significant_hits_annotated()` | mmc6 | Gene–metabolite pairs passing significance; includes FBA reactant status, pathway, operon annotation | 1385 × 29 |
| `load_significant_hits_ms2()` | mmc7 | Extends mmc6 with LC-MS/MS spectra, SMILES, retention times, QC flags | 1256 × 54 |
| `load_significant_fold_changes()` | mmc8 | Compact significant hits: one row per (gene, metabolite, ionisation mode) | 9462 × 10 |
| `load_nonannotated_features()` | mmc9 | SIRIUS/CSI:FingerID annotations for m/z features not matched to iML1515 | 52 × 27 |
| `load_metabolite_reference()` | mmc10 | **Authors' Rapp → BiGG mapping**: `Abbreviation` (Rapp), `BIGG`, `Metabolite`, `KEGG`, `Monoisotopic_Mass`, `Formula` | 802 × 6 |
| `load_gene_substrates()` | mmc11 | Gene → substrate metabolites from iML1515 stoichiometry | 7683 × 9 |
| `load_pathway_positions()` | mmc12 | Gene ordinal position within EcoCyc/BioCyc pathways | 3340 × 11 |
| `load_pathway_metabolites()` | mmc13 | Extends mmc12 with substrate/product metabolites at each pathway position | 11851 × 18 |

### Key design notes

**Metabolite abbreviations** — the `Abbrev` column in mmc5 encodes both isobaric
ambiguity (pairs of same-mass metabolites joined by `-`, e.g. `ac-gcald`) and MS
adduct notation (e.g. `[M+H]+`). Use `load_metabolite_reference()` (mmc10) as the
authoritative Rapp → BiGG mapping rather than stripping adducts manually: the
authors pre-resolved stereochemistry notation differences and isobaric groupings
for all 802 detected metabolites, achieving 801/802 coverage against iML1515.

**Sample ID format** — mmc5 sample columns follow `{gene}_{Rn}_{msAVxxx}_{Bn}`.
`load_endpoint_od()` returns matching `Sample_ID` values for joining metadata.

**Significance tables** — mmc6/mmc7 are feature-level (one row per detected ion);
mmc8 is the compact per-(gene, metabolite) summary. For most analyses, start
with mmc8.

---

## iML1515 model loaders (`load_bigg.py`)

Model file: `iML1515.json` (3.0 MB, downloaded from BiGG).
Stats: **1877 metabolites · 2712 reactions · 1516 genes**.

| Function | Returns | Index | Shape |
|---|---|---|---|
| `bigg_metabolite_df()` | One row per compartment-specific metabolite | full BiGG id (`10fthf_c`) | 1877 × 8 |
| `bigg_reaction_df()` | One row per reaction | reaction id (`PFK`) | 2712 × 10 |
| `bigg_gene_df()` | One row per gene | gene **name** (`pfkA`) | 1516 × 4 |
| `bigg_gene_reaction_df()` | Long gene→reaction associations | — | ~9000 × 4 |
| `stoichiometry_long()` | Tidy (metabolite, reaction, coefficient) | — | ~10700 × 4 |
| `stoichiometry_wide()` | Metabolite × reaction matrix | full metabolite id | 1877 × 2712 |

### Key design choices

**Metabolite index** — full id with compartment (`10fthf_c`, `10fthf_e`, `10fthf_p`)
preserves the network topology. The `base_id` column strips the compartment suffix
and is the join key for Rapp data.

**Gene index** — gene *name* (`pfkA`, `glyA`, …). Names are unique in iML1515.
b-numbers are kept in column `b_number`. Note: transcription factors (e.g. `arcA`)
are not in iML1515 — it only contains enzymes with metabolic reactions.

**Stoichiometry sign convention** — negative coefficient = substrate (consumed);
positive = product (produced). This is standard BiGG / COBRA convention.

### Rapp ↔ iML1515 metabolite overlap

```python
import load, load_bigg
met_mat = load.load_metabolomics_matrix()          # mmc5
rapp_base = load_bigg.rapp_to_bigg_index(met_mat['Abbrev'])  # strip adducts
mdf = load_bigg.bigg_metabolite_df()
overlap_mask = rapp_base.isin(mdf['base_id'])
```

`rapp_to_bigg_index()` strips the MS adduct notation (`[M+H]+`, `[M-H]-`, etc.)
from mmc5 Abbrev values. Isobaric entries joined by `-` are left intact as a
group; call `.str.split('-').explode()` to get one metabolite per row.
