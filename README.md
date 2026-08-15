# Zhao Hanqing TCM Prescription Dataset 2025 — Analysis Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21951692.svg)](https://doi.org/10.5281/zenodo.21951692)

Analysis code accompanying the de-identified outpatient prescription dataset of
**Professor Zhao Hanqing** (National TCM Master), Heniantang Clinic, January–December 2025.

## Dataset

**DOI**: https://doi.org/10.5281/zenodo.21951692 (v1.1, CC BY 4.0)

- 335 prescriptions, 5,951 herb-level records, 122 de-identified patients
- 438 unique herb strings with doses; 91 disease labels; 102 syndrome-pattern labels
- Privacy-generalised release: age bands (10–29 … 80–89), half-year visit periods, minimum k-anonymity = 3

## Scripts

| Script | Analysis |
|---|---|
| `01_frequency_analysis.py` | Herb frequency, dose statistics, cohort characteristics |
| `02_association_rules.py` | Association-rule mining (support/confidence/lift) |
| `03_cluster_factor_analysis.py` | Hierarchical clustering, factor analysis |
| `04_innovation_analysis.py` | Novelty metrics, co-prescription network, seasonal profiling |
| `prepare_public_dataset.py` | Dataset preparation and de-identification pipeline |

## Requirements

Python 3.11+, pandas, scikit-learn, mlxtend, networkx, matplotlib, seaborn, scipy.

## Usage

```bash
python 01_frequency_analysis.py --help
```

Each script reads the Zenodo CSV tables (download from the DOI link above) and
writes reproducible outputs to a local `results/` directory.

## Citation

Zhao, H. (2026). De-identified outpatient prescription dataset of a national TCM
master (Zhao Hanqing, 2025): 335 prescriptions with 5,951 herb-level records
(Version v1.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21951692

## License

Code: MIT License. Data (via Zenodo): CC BY 4.0.
