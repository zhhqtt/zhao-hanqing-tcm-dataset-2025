# Zhao Hanqing TCM Prescription Dataset 2025 — Analysis Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21947836.svg)](https://doi.org/10.5281/zenodo.21947836)

Analysis code accompanying the de-identified outpatient prescription dataset of
**Prof. ZHAO Hanqing M.D., Ph.D.** (renowned TCM practitioner), Beijing Heniantang Hospital, January–December 2025.

## Dataset

**DOI**: https://doi.org/10.5281/zenodo.21947836 (concept DOI; **v1.3.1 is the current release**, version DOI 10.5281/zenodo.21973162; CC BY 4.0)

- 335 prescriptions, 5,951 herb-level records, 122 de-identified patients
- 438 unique herb strings with doses
- **90 substantive disease labels** (91 values including "Not recorded") and **101 substantive syndrome-pattern labels** (102 values including "Not recorded"). The Chinese source contains 91 substantive disease labels: 腰痛 and 背痛 are merged into "Back pain" in the English translation layer.
- **Schema (v1.3.1, verified against the release archive)**:
  - Herb-level table `ZhaoHanqing_TCM_Prescriptions_2025.csv` — 5,951 rows × **17 columns**: `prescription_id`, `patient_id`, `visit_type`, `patient_sex`, `herb_name_pinyin`, `herb_name_chinese`, `herb_latin_name`, `dose_g`, `syndrome_pattern`, `disease_name`, `season`, `department`, `patient_age_group`, `visit_half_year`, `visit_solar_term`, `administration_route`, `dosing_frequency`.
  - Summary table `ZhaoHanqing_Prescription_Summary_2025.csv` — 335 rows × **15 columns**: the same prescription-level fields plus `total_herbs` and `herb_list_pinyin` (comma-separated Pinyin list); herb-level fields omitted.
  - The v1.2 additions `visit_solar_term` (Chinese solar term/jieqi of the prescription date, 24 standard English names), `administration_route`, and `dosing_frequency` are retained unchanged in v1.3/v1.3.1; likewise the privacy-generalised `patient_age_group` (7 bands: 10–29 … 80–89) and `visit_half_year` (2025-H1 / 2025-H2). Releases v1.3 and v1.3.1 were metadata-only / record-hygiene revisions — **no data changes**; all tables are cell-identical to v1.2.
- **Privacy**: age bands and half-year visit periods give minimum k-anonymity = 3 over (age band × sex × half-year; 24 equivalence classes, max k = 37). The `visit_solar_term` field (added in v1.2) narrows the visit window from half-year (~182 d) to ~15 days: over (sex × age band × solar term) there are **161 equivalence classes with minimum k = 1, maximum k = 8, and 116 classes with k < 3** — a deliberate release decision; the re-identification risk assessment is documented in the dataset's BUILD_REPORT and README (*Privacy* section, distributed via Zenodo).

## Scripts

| Script | Analysis |
|---|---|
| `01_frequency_analysis.py` | Herb frequency, dose statistics, cohort characteristics |
| `02_association_rules.py` | Association-rule mining (support/confidence/lift) |
| `03_cluster_factor_analysis.py` | Hierarchical clustering, factor analysis |
| `04_innovation_analysis.py` | Novelty metrics, co-prescription network, seasonal profiling |
| `prepare_public_dataset.py` | Dataset preparation and de-identification pipeline (v1.0/v1.1) |
| `prepare_v1_2_solar.py` | v1.2 build: solar-term assignment, audited `administration_route` / `dosing_frequency`, solar-term k-anonymity diagnostic, release zip |

`prepare_v1_2_solar.py` is the exact build script of the data tables and is published as a
provenance record; it contains local input paths (`BASE = ...`) to be adapted, and reads the
private cleaned source plus the v1.1 release folder. Because releases v1.3 and v1.3.1 made no
data changes (metadata-only / record hygiene), this script remains the build script of the
current v1.3.1 data tables. The public data tables themselves are distributed via Zenodo only
(this repository contains code and documentation, no data files).

## Requirements

**Python 3.12** (validated on 3.12.3). See `requirements.txt` for the pinned versions used to
validate the scripts:

pandas, NumPy, SciPy, scikit-learn, NetworkX, Matplotlib,
mlxtend (Apriori / association rules in `02_association_rules.py`),
seaborn (figure-generation scripts), and
python-louvain (`import community` in `02_association_rules.py`).

## Usage

```bash
python 01_frequency_analysis.py --help
```

Each script reads the Zenodo CSV tables (download from the DOI link above) and
writes reproducible outputs to a local `results/` directory.

## Citation

ZHAO, H. (2026). De-identified outpatient prescription dataset of a renowned TCM practitioner
(Prof. ZHAO Hanqing M.D., Ph.D., 2025): 335 prescriptions with 5,951 herb-level records
(Version v1.3.1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21973162

## License

Code: MIT License. Data (via Zenodo): CC BY 4.0.
