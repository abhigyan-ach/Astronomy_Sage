# AstroProject - Symbiotic Star Candidate Detection

This repository contains a pipeline for identifying symbiotic star candidates from multi-survey astronomical data, combining data from SkyMapper, Gaia, 2MASS, and AllWISE surveys.

## Project Overview

The project applies a systematic approach to:
1. Query and cross-match multi-wavelength astronomical data
2. Apply quality cuts and distance corrections
3. Compute photometric colors across multiple filters
4. Merge photometry with imaging data
5. Perform anomaly detection using Variational Autoencoders (VAE)

## Data Pipeline Sequence

To reproduce the complete data pipeline, run the following scripts **in order**:

### 1. **`query.py`** (Data Acquisition)
This script queries the SkyMapper DR2 archive and performs cross-matches with:
- **SkyMapper DR2**: u, v, g, r, i, z photometry
- **Gaia DR2**: Parallax, proper motion, astrometric data
- **2MASS**: J, H, K near-infrared photometry
- **AllWISE**: W1, W2 mid-infrared photometry

**Key Selection Criteria:**
- Cross-match tolerances: 2MASS < 2", Gaia < 2", AllWISE < 3"
- High-quality 2MASS photometry (`ph_qual = 'AAA'`)
- Point source selection (`class_star > 0.9`)
- Photometric flags filtering
- Color criterion: J-K > 0.85 (red color characteristic of symbiotic stars)
- Brightness limit: J < 14.0
- Parallax constraints: 0 < parallax < 100 mas

**Output:** `Dr2_table.csv`

---

### 2. **`apply_cuts.py`** (Quality Filtering & Distance Corrections)
Applies additional quality cuts and merges Bailer-Jones distance estimates from Gaia.

**Key Functions:**
- `gaia_bailer_jones_distances()`: Queries Gaia DR2 geometric distances
- `cut3()`: Applies extinction-corrected color cut: (J-K - 0.413×E(B-V)) > 0.85
- `cut4()`: Applies distance-corrected magnitude cut using Bailer-Jones distances

**Outputs:**
- `bailer_jones_distances.csv`: Distance estimates with uncertainties
- `filtered_data_ebmv_cut_Dr2_table_after_bailer_jones.csv`: Quality-filtered catalog

---

### 3. **`compute_all_colors.py`** (Color Computation)
Computes all possible color combinations from photometric data across 6 filters (u, v, g, r, i, z).

**Key Features:**
- Computes nightly magnitudes using weighted averages
- Propagates measurement uncertainties
- Calculates **all 30 permutations** of color indices (e.g., u-g, u-v, v-u, g-r, etc.)
- Uses inverse-variance weighting for optimal color estimates

**Method:**
1. Groups photometry by object and night
2. Computes weighted mean magnitude per night per filter
3. Calculates color differences for all filter pairs
4. Aggregates nightly colors into final per-object colors

**Output:** `adrian_all_weighted_colors.csv`

---

### 4. **`merge_image_and_photometry.py`** (Data Consolidation)
Merges the computed colors with the filtered photometric catalog.

**Process:**
- Inner join on `object_id` between color catalog and DR2 filtered data
- Ensures only objects with complete photometry and computed colors are retained
- Reports memory usage and final sample size

**Output:** `merged_df_w3w4_final_all_colors.csv`

---

## Variational Autoencoder (VAE) Analysis

### **`vae/variational_autoencoder.py`** (Anomaly Detection)
Uses a Variational Autoencoder to identify anomalous objects (potential symbiotic star candidates) based on their photometric properties.

**Purpose:**
Variational Autoencoders learn to encode high-dimensional photometric data into a low-dimensional latent space and reconstruct the input. Objects that are poorly reconstructed (high reconstruction loss) are flagged as anomalies, potentially indicating rare or unusual objects like symbiotic stars.

**Input Features:**
- 9 photometric magnitudes: u, v, g, r, i, z (SkyMapper), J, H, K (2MASS)
- Data is median-imputed and standardized

**Architecture:**
- **Encoder**: Input (9D) → 16 hidden units → 3D latent space (mean + log-variance)
- **Decoder**: 3D latent space → 16 hidden units → Output (9D)
- **Loss Function**: Reconstruction loss (MSE) + KL divergence

**Training:**
- 100 epochs with batch size 64
- Adam optimizer (learning rate 1e-3)

**Anomaly Detection:**
- Computes reconstruction loss for each object
- Flags top 2% highest losses as anomalies
- Anomalous objects are likely unusual/rare sources

**Outputs:**
- `anomalous_object_ids.txt`: List of flagged object IDs
- `vae_reconstruction_losses.csv`: Reconstruction loss for all objects
- Visualization plots:
  - `vae_loss_histogram.png`: Loss distribution with anomaly threshold
  - `vae_loss_boxplot.png`: Statistical summary of losses
  - `vae_loss_scatter.png`: Loss per object with anomalies highlighted

**Interpretation:**
Objects with high reconstruction loss deviate significantly from the "typical" photometric properties learned by the VAE. These are strong candidates for follow-up as potential symbiotic stars or other rare variable objects.

---

## Additional Scripts

- **`adrian_compute_all_colors.py`**: Alternative color computation implementation (similar to `compute_all_colors.py`)
- **`pca_analysis.py`**: Principal Component Analysis for dimensionality reduction
- **`match_known_symbiotics.py`**: Cross-matches candidates with known symbiotic star catalogs
- **`get_bailerjones_distances.py`**: Standalone script for querying Bailer-Jones distances

---

## Requirements

### Python Dependencies
```
pandas
numpy
matplotlib
seaborn
astropy
astroquery
pyvo
torch (PyTorch)
scikit-learn
mwdust
healpy
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn astropy astroquery pyvo torch scikit-learn mwdust healpy
```

---

## Quick Start

Run the complete pipeline:

```bash
# Step 1: Query and acquire data
python query.py

# Step 2: Apply quality cuts and distance corrections
python apply_cuts.py

# Step 3: Compute photometric colors
python compute_all_colors.py

# Step 4: Merge colors with photometry
python merge_image_and_photometry.py

# Step 5 (Optional): Run VAE anomaly detection
python vae/variational_autoencoder.py
```

---

## Data Products

### Intermediate Files
- `Dr2_table.csv`: Initial cross-matched catalog
- `bailer_jones_distances.csv`: Distance estimates
- `filtered_data_ebmv_cut_Dr2_table_after_bailer_jones.csv`: Quality-filtered list of luminous red objects
- `adrian_all_weighted_colors.csv`: Computed color indices from the list of luminous red objects

### Final Data Product
- `merged_df_w3w4_final_all_colors.csv`: Complete catalog with photometry and colors

### VAE Outputs
- `anomalous_object_ids.txt`: Candidate symbiotic stars
- `vae_reconstruction_losses.csv`: Anomaly scores for all objects

---

## Scientific Background

**Symbiotic stars** are interacting binary systems consisting of:
- A red giant (cool, evolved star)
- A hot compact companion (white dwarf, neutron star, or main-sequence star)



This pipeline identifies candidates by:
1. Selecting red, point-like objects with good photometry
2. Computing multi-band colors to capture the SED
3. Using VAE to identify objects with unusual color combinations

---


Usage:
To get the data you need to run the following files in unison:
SQL_Query.py-->apply_cuts.py-->compute_all_colors.py->merge_all_colors.py

## Citation

If you use this pipeline or data products, please cite:
- **SkyMapper**: [Onken et al. 2019, PASA, 36, e033](https://ui.adsabs.harvard.edu/abs/2019PASA...36...33O)
- **Gaia DR2**: [Gaia Collaboration et al. 2018, A&A, 616, A1](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A...1G)
- **Bailer-Jones Distances**: [Bailer-Jones et al. 2018, AJ, 156, 58](https://ui.adsabs.harvard.edu/abs/2018AJ....156...58B)

---

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

---

## License

This project is distributed under the MIT License. See `LICENSE` file for details.
