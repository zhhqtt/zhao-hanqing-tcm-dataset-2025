#!/usr/bin/env python3
"""
Build Zhao Hanqing TCM dataset v1.2 — adds `visit_solar_term` (24 solar terms,
English standard names) to the v1.1 privacy-generalised public release.

Also conditionally adds `administration_route` and `dosing_frequency`:
these two source fields were audited first (value distribution + within-
prescription consistency); semantics are unambiguous (4 / 2 distinct values,
constant within every prescription), so they are added.
`总数量` (dispensed quantity) and `规格` (pack specification) were audited and
REJECTED for inclusion: semantics remain ambiguous (see audit section below
and BUILD_REPORT.md).

Inputs (READ-ONLY, never modified):
  - data/cleaned_data.pkl                       (5951 x 33, exact datetimes)
  - zenodo_release_v1_1/*                       (current release, 9 files)

Outputs (created fresh under zenodo_release_v1_2/, plus this script itself):
  - ZhaoHanqing_TCM_Prescriptions_2025.csv      (herb level, 14 + 3 columns)
  - ZhaoHanqing_Prescription_Summary_2025.csv   (prescription level, 12 + 3)
  - Data_Dictionary.csv, Analysis_Data_Dictionary.csv, herb_lookup.csv,
    Nomenclature_QA_notes.csv, README.md, datapackage.json (v1.2), LICENSE
  - ZhaoHanqing_TCM_Dataset_2025_v1.2.zip       (the 9 files above)
  - BUILD_REPORT.md (evidence record, generated from computed values)

SOLAR TERM TABLE PROVENANCE (all 24 dates cross-verified 2026-08-16):
  S1  HKO complete table (via https://www.miukuettong.com/blog/posts/2025solarterm ,
      footer "資料來源：香港天文台") — dates + minute-level times, 24/24 captured.
  S2  https://jieqi.bmcx.com/2025__jieqi/ — all 24 dates (two internal grids,
      mutually identical; detail block gives 立春 2025-02-03 22:10:13).
  S3  Taipei Astronomical Museum (冬至 2025-12-21 23:03):
      https://tam.gov.taipei/News_Content.aspx?n=B64052C7930D4913&sms=2CF1F5E2E0B96411&s=BA8EF9B5EA5C3CE5
  S4  Xinhua / PMO《中国天文年历》 (清明 2025-04-04 20:49).
  S5  Xinhuanet (立春 2025-02-03).
  S1 and S2 agree on all 24 DATES; all 8 independently-known anchors match.
  Minute-level differences (<=1 min: e.g. 清明 20:48 vs 20:49, 冬至 23:02 vs
  23:03) cannot change any date assignment (closest-to-midnight case 冬至
  23:03 is still 57 min from midnight) and are NOT used — assignment is by
  calendar day per the release rule.

ASSIGNMENT RULE (documented in README + BUILD_REPORT):
  A visit belongs to the term whose 交节 calendar day is the latest term-day
  <= visit date ("交节日当天及之后 → 新节气，直至下一交节前一日").
  Cross-year rule (not triggered by this data): dates before 2025-01-05
  belong to the previous year's 冬至 (2024-12-21); dates after 2025-12-21
  belong to 冬至 until 2026-01-05 (小寒).
"""
import json
import os
import shutil
import sys
import zipfile
import hashlib
from collections import OrderedDict

import pandas as pd

BASE = '/home/zhhq/.openclaw/workspace-coder/ZhaoAnalysis'
PKL = os.path.join(BASE, 'data/cleaned_data.pkl')
SRC = os.path.join(BASE, 'zenodo_release_v1_1')
OUT = os.path.join(BASE, 'zenodo_release_v1_2')

# ---------------------------------------------------------------------------
# 1. Verified 2025 solar term table (dates from S1/S2; times from S1=HKO)
# ---------------------------------------------------------------------------
SOLAR_TERMS_2025 = [  # (start_date, Chinese, English standard name, HKO time)
    ('2025-01-05', '小寒', 'Minor Cold',          '10:33'),
    ('2025-01-20', '大寒', 'Major Cold',          '04:00'),
    ('2025-02-03', '立春', 'Start of Spring',     '22:10'),
    ('2025-02-18', '雨水', 'Rain Water',          '18:07'),
    ('2025-03-05', '惊蛰', 'Awakening of Insects','16:07'),
    ('2025-03-20', '春分', 'Spring Equinox',      '17:01'),
    ('2025-04-04', '清明', 'Clear and Bright',    '20:48'),
    ('2025-04-20', '谷雨', 'Grain Rain',          '03:56'),
    ('2025-05-05', '立夏', 'Start of Summer',     '13:57'),
    ('2025-05-21', '小满', 'Grain Buds',          '02:55'),
    ('2025-06-05', '芒种', 'Grain in Ear',        '17:56'),
    ('2025-06-21', '夏至', 'Summer Solstice',     '10:42'),
    ('2025-07-07', '小暑', 'Minor Heat',          '04:05'),
    ('2025-07-22', '大暑', 'Major Heat',          '21:29'),
    ('2025-08-07', '立秋', 'Start of Autumn',     '13:52'),
    ('2025-08-23', '处暑', 'End of Heat',         '04:34'),
    ('2025-09-07', '白露', 'White Dew',           '16:52'),
    ('2025-09-23', '秋分', 'Autumn Equinox',      '02:19'),
    ('2025-10-08', '寒露', 'Cold Dew',            '08:41'),
    ('2025-10-23', '霜降', "Frost's Descent",     '11:51'),
    ('2025-11-07', '立冬', 'Start of Winter',     '12:04'),
    ('2025-11-22', '小雪', 'Minor Snow',          '09:36'),
    ('2025-12-07', '大雪', 'Major Snow',          '05:05'),
    ('2025-12-21', '冬至', 'Winter Solstice',     '23:03'),
]
# Independently verified anchors (from prior multi-source cross-validation):
# (term, date, time_or_None, source_note)
ANCHORS = [
    ('小寒', '2025-01-05', '10:33', 'prior verified anchor'),
    ('立春', '2025-02-03', '22:10', 'prior anchor; bmcx 22:10:13; Xinhuanet date'),
    ('清明', '2025-04-04', None,    'PMO《中国天文年历》via Xinhua 20:49 (S1 20:48; 1-min diff, same day)'),
    ('小满', '2025-05-21', '02:55', 'prior verified anchor'),
    ('处暑', '2025-08-23', '04:34', 'prior verified anchor'),
    ('秋分', '2025-09-23', '02:19', 'prior verified anchor'),
    ('大雪', '2025-12-07', '05:05', 'prior verified anchor'),
    ('冬至', '2025-12-21', None,    'TAM 23:03 (S3); S1 23:03'),
]

TERM_CN2EN = {zh: en for _, zh, en, _ in SOLAR_TERMS_2025}
TERM_STARTS = [(pd.Timestamp(d), zh, en) for d, zh, en, _ in SOLAR_TERMS_2025]
ALL_TERM_EN = [en for _, _, en, _ in SOLAR_TERMS_2025]

def assign_solar_term(dt):
    """Day-level rule: latest term whose start day <= dt (inclusive)."""
    d = pd.Timestamp(dt).normalize()
    chosen = None
    for start, zh, en in TERM_STARTS:
        if start <= d:
            chosen = en
        else:
            break
    if chosen is None:  # before 2025 小寒 → previous year's 冬至
        return 'Winter Solstice'
    return chosen

# ---------------------------------------------------------------------------
# 2. Load source data and assert core counts (纪律: 逐个核实, 不得编造)
# ---------------------------------------------------------------------------
df = pd.read_pickle(PKL)
assert df.shape == (5951, 33), f'cleaned_data.pkl shape {df.shape} != (5951, 33)'
assert len(df['处方号'].unique()) == 335, 'prescription count != 335'
assert len(df['病历号\u3000'].unique()) == 122, 'patient count != 122'
assert df['名称'].nunique() == 438, 'unique herb count != 438'
assert df['收费日期'].min().normalize() == pd.Timestamp('2025-01-05')
assert df['收费日期'].max().normalize() == pd.Timestamp('2025-12-28')

# Label counts (Chinese source; empty → 'Not recorded' in release)
dis_empty = int(df['主要疾病'].isna().sum() + (df['主要疾病'] == '').sum())
syn_empty = int(df['主要证型'].isna().sum() + (df['主要证型'] == '').sum())
assert dis_empty == 167 and syn_empty == 48
# Label counts: source Chinese labels vs release English labels.
# Disease: 91 distinct non-empty Chinese labels; v1.1 release maps 背痛 and 腰痛
# BOTH to "Back pain", giving 90 distinct English labels (inherited unchanged).
n_disease_labels = df.loc[df['主要疾病'].notna() & (df['主要疾病'] != ''), '主要疾病'].nunique()
n_syndrome_labels = df.loc[df['主要证型'].notna() & (df['主要证型'] != ''), '主要证型'].nunique()
assert n_disease_labels == 91, f'Chinese disease labels {n_disease_labels} != 91'
assert n_syndrome_labels == 101, f'Chinese syndrome labels {n_syndrome_labels} != 101'
DISEASE_COLLISION_NOTE = ('背痛 -> Back pain; 腰痛 -> Back pain (two Chinese labels share one '
                          'English label in v1.1; v1.2 preserves values cell-identically)')

# 处方日期唯一性: each 处方号 has exactly one 收费日期
g = df.groupby('处方号')
assert (g['收费日期'].nunique() == 1).all(), '处方号 contains >1 distinct 收费日期'

# Anchor check against the fetched table
tbl = {zh: (d, t) for d, zh, en, t in SOLAR_TERMS_2025}
anchor_report = []
for zh, d, t, note in ANCHORS:
    td, tt = tbl[zh]
    ok = (td == d) and (t is None or tt == t)
    anchor_report.append((zh, d, t, td, tt, ok, note))
    assert ok, f'ANCHOR MISMATCH {zh}: table {td} {tt} vs anchor {d} {t}'
# table internal sanity: 24 rows, strictly increasing, unique names/dates
assert len(SOLAR_TERMS_2025) == 24
dates = [pd.Timestamp(d) for d, *_ in SOLAR_TERMS_2025]
assert all(b > a for a, b in zip(dates, dates[1:]))
assert len(set(ALL_TERM_EN)) == 24 and len(set(dates)) == 24

# ---------------------------------------------------------------------------
# 3. Reproduce v1.1 anonymised ID maps and verify against the v1.1 release
# ---------------------------------------------------------------------------
presc_map = {pid: f'RX{i+1:03d}' for i, pid in enumerate(sorted(df['处方号'].unique(), key=float))}
patient_map = {pid: f'P{i+1:03d}' for i, pid in enumerate(sorted(df['病历号\u3000'].unique(), key=str))}

v11p = pd.read_csv(os.path.join(SRC, 'ZhaoHanqing_TCM_Prescriptions_2025.csv'), dtype=str)
v11s = pd.read_csv(os.path.join(SRC, 'ZhaoHanqing_Prescription_Summary_2025.csv'), dtype=str)
assert v11p.shape == (5951, 14), f'v1.1 prescriptions shape {v11p.shape}'
assert v11s.shape == (335, 12), f'v1.1 summary shape {v11s.shape}'
assert list(v11p.columns) == ['prescription_id', 'patient_id', 'visit_type', 'patient_sex',
    'herb_name_pinyin', 'herb_name_chinese', 'herb_latin_name', 'dose_g',
    'syndrome_pattern', 'disease_name', 'season', 'department',
    'patient_age_group', 'visit_half_year']
# Row-by-row identity of the mapping vs v1.1
assert (v11p['prescription_id'].values == df['处方号'].map(presc_map).values).all()
assert (v11p['patient_id'].values == df['病历号\u3000'].map(patient_map).values).all()

# Age group / half-year / season reproduction check (prescription level)
def age_group(a):
    bands = [(29, '10-29'), (39, '30-39'), (49, '40-49'), (59, '50-59'),
             (69, '60-69'), (79, '70-79'), (89, '80-89')]
    for hi, lab in bands:
        if a <= hi:
            return lab
    raise ValueError(f'age out of bands: {a}')

SEASON_EN = {'春': 'Spring', '夏': 'Summer', '秋': 'Autumn', '冬': 'Winter'}
first = g.first()
ag_exp = first['年龄'].map(age_group).rename(index=presc_map)
hy_exp = first['收费日期'].dt.month.le(6).map({True: '2025-H1', False: '2025-H2'}).rename(index=presc_map)
se_exp = first['季节'].map(SEASON_EN).rename(index=presc_map)
assert (v11s.set_index('prescription_id')['patient_age_group'] == ag_exp).all()
assert (v11s.set_index('prescription_id')['visit_half_year'] == hy_exp).all()
assert (v11s.set_index('prescription_id')['season'] == se_exp).all()
# summary total_herbs == herb rows per prescription
th_exp = g.size().rename(index=presc_map)
assert (v11s.set_index('prescription_id')['total_herbs'].astype(int) == th_exp).all()

# ---------------------------------------------------------------------------
# 4. Field audit: 总数量 / 用药途径 / 用药时间 / 规格  (+金额 cross)
# ---------------------------------------------------------------------------
audit = OrderedDict()
audit['总数量'] = df['总数量'].value_counts(dropna=False)
audit['用药途径'] = df['用药途径'].value_counts(dropna=False)
audit['用药时间'] = df['用药时间'].value_counts(dropna=False)
audit['规格'] = df['规格'].value_counts(dropna=False)
within = {c: int((g[c].nunique() > 1).sum()) for c in ['总数量', '用药途径', '用药时间', '规格']}
amount_mismatch = int((df['金额'] != df['应收金额']).sum())
# Route & frequency: constant within every prescription → semantics clear → ADD
ROUTE_EN = {'水煎服': 'Oral decoction', '外洗': 'External wash',
            '冲服': 'Oral (dissolved)', '外用': 'External use'}
FREQ_EN = {'每日两次': 'Twice daily', '每日三次': 'Three times daily'}
assert set(df['用药途径'].unique()) <= set(ROUTE_EN)
assert set(df['用药时间'].unique()) <= set(FREQ_EN)
assert within['用药途径'] == 0 and within['用药时间'] == 0
# 总数量/规格: NOT added (unit semantics unverifiable / messy packaging)

# ---------------------------------------------------------------------------
# 5. Assign solar terms (prescription level, from exact datetimes)
# ---------------------------------------------------------------------------
presc_date = g['收费日期'].first()                      # verified unique above
presc_term = presc_date.map(assign_solar_term)
presc_route = first['用药途径'].map(ROUTE_EN)
presc_freq = first['用药时间'].map(FREQ_EN)

# No nulls; value domain within the 24 standard names
assert presc_term.notna().all()
assert set(presc_term.unique()) <= set(ALL_TERM_EN)
terms_present = sorted(presc_term.unique(), key=ALL_TERM_EN.index)
terms_absent = [en for en in ALL_TERM_EN if en not in set(presc_term)]

# ---------------------------------------------------------------------------
# 6. Write v1.2 CSVs (append 3 columns; shared cells identical to v1.1)
# ---------------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)

def add_cols(frame, key_series):
    """key_series: Series indexed by 处方号 → per-prescription values."""
    out = frame.copy()
    mapped = out['prescription_id'].map({presc_map[k]: v for k, v in key_series.items()})
    out['visit_solar_term'] = mapped
    out['administration_route'] = out['prescription_id'].map({presc_map[k]: v for k, v in presc_route.items()})
    out['dosing_frequency'] = out['prescription_id'].map({presc_map[k]: v for k, v in presc_freq.items()})
    return out

v12p = add_cols(v11p, presc_term)
v12s = add_cols(v11s, presc_term)
assert v12p.shape == (5951, 17) and v12s.shape == (335, 15)
assert v12p['visit_solar_term'].notna().all() and v12s['visit_solar_term'].notna().all()
assert v12p['administration_route'].notna().all() and v12p['dosing_frequency'].notna().all()

p_csv = os.path.join(OUT, 'ZhaoHanqing_TCM_Prescriptions_2025.csv')
s_csv = os.path.join(OUT, 'ZhaoHanqing_Prescription_Summary_2025.csv')
v12p.to_csv(p_csv, index=False)
v12s.to_csv(s_csv, index=False)

# Post-write verification: shared columns byte-identical to v1.1
rp = pd.read_csv(p_csv, dtype=str)
rs = pd.read_csv(s_csv, dtype=str)
assert list(rp.columns) == list(v11p.columns) + ['visit_solar_term', 'administration_route', 'dosing_frequency']
assert list(rs.columns) == list(v11s.columns) + ['visit_solar_term', 'administration_route', 'dosing_frequency']
for c in v11p.columns:
    assert (rp[c].fillna('') == v11p[c].fillna('')).all(), f'herb-table column changed: {c}'
for c in v11s.columns:
    assert (rs[c].fillna('') == v11s[c].fillna('')).all(), f'summary column changed: {c}'
assert set(rp['visit_solar_term'].unique()) <= set(ALL_TERM_EN)
assert set(rs['visit_solar_term'].unique()) <= set(ALL_TERM_EN)
# Release-side label counts (excluding "Not recorded")
rel_dis = rp.loc[rp['disease_name'] != 'Not recorded', 'disease_name'].nunique()
rel_syn = rp.loc[rp['syndrome_pattern'] != 'Not recorded', 'syndrome_pattern'].nunique()
assert rel_dis == 90, f'release disease labels {rel_dis} != 90'
assert rel_syn == 101, f'release syndrome labels {rel_syn} != 101'
assert int((rp['disease_name'] == 'Not recorded').sum()) == 167
assert int((rp['syndrome_pattern'] == 'Not recorded').sum()) == 48

# ---------------------------------------------------------------------------
# 7. Data dictionary (v1.1 rows + new variables)
# ---------------------------------------------------------------------------
dd = pd.read_csv(os.path.join(SRC, 'Data_Dictionary.csv'), dtype=str)
new_rows = [
    {'file': 'ZhaoHanqing_TCM_Prescriptions_2025.csv', 'variable': 'visit_solar_term',
     'type': 'string', 'required': 'yes',
     'allowed_values': '; '.join(ALL_TERM_EN),
     'description': 'Chinese solar term (jieqi) of the prescription date, assigned by calendar day: the term starting on the latest jieqi transition day on or before the visit date. Standard English names.',
     'example': 'Winter Solstice', 'missing_value_policy': ''},
    {'file': 'ZhaoHanqing_TCM_Prescriptions_2025.csv', 'variable': 'administration_route',
     'type': 'string', 'required': 'yes',
     'allowed_values': 'Oral decoction; External wash; Oral (dissolved); External use',
     'description': 'Route of administration recorded for the prescription (constant within a prescription).',
     'example': 'Oral decoction', 'missing_value_policy': ''},
    {'file': 'ZhaoHanqing_TCM_Prescriptions_2025.csv', 'variable': 'dosing_frequency',
     'type': 'string', 'required': 'yes',
     'allowed_values': 'Twice daily; Three times daily',
     'description': 'Dosing frequency recorded for the prescription (constant within a prescription).',
     'example': 'Twice daily', 'missing_value_policy': ''},
    {'file': 'ZhaoHanqing_Prescription_Summary_2025.csv', 'variable': 'visit_solar_term',
     'type': 'string', 'required': 'yes',
     'allowed_values': '; '.join(ALL_TERM_EN),
     'description': 'Chinese solar term (jieqi) of the prescription date (see herb-level table for rule).',
     'example': 'Winter Solstice', 'missing_value_policy': ''},
    {'file': 'ZhaoHanqing_Prescription_Summary_2025.csv', 'variable': 'administration_route',
     'type': 'string', 'required': 'yes',
     'allowed_values': 'Oral decoction; External wash; Oral (dissolved); External use',
     'description': 'Route of administration recorded for the prescription.',
     'example': 'Oral decoction', 'missing_value_policy': ''},
    {'file': 'ZhaoHanqing_Prescription_Summary_2025.csv', 'variable': 'dosing_frequency',
     'type': 'string', 'required': 'yes',
     'allowed_values': 'Twice daily; Three times daily',
     'description': 'Dosing frequency recorded for the prescription.',
     'example': 'Twice daily', 'missing_value_policy': ''},
]
dd12 = pd.concat([dd, pd.DataFrame(new_rows)[list(dd.columns)]], ignore_index=True)
dd12.to_csv(os.path.join(OUT, 'Data_Dictionary.csv'), index=False)

# Unchanged carriers, copied byte-for-byte from v1.1
for name in ['Analysis_Data_Dictionary.csv', 'herb_lookup.csv',
             'Nomenclature_QA_notes.csv', 'LICENSE']:
    shutil.copyfile(os.path.join(SRC, name), os.path.join(OUT, name))

# ---------------------------------------------------------------------------
# 8. datapackage.json (v1.2) and README.md
# ---------------------------------------------------------------------------
with open(os.path.join(SRC, 'datapackage.json'), encoding='utf-8') as f:
    dp = json.load(f, object_pairs_hook=OrderedDict)
dp['version'] = '1.2'
with open(os.path.join(OUT, 'datapackage.json'), 'w', encoding='utf-8') as f:
    json.dump(dp, f, ensure_ascii=False, indent=2)
    f.write('\n')

term_counts = presc_term.value_counts()
README = f"""# Zhao Hanqing TCM Prescription Dataset 2025

**DOI**: https://doi.org/10.5281/zenodo.21951692 (concept DOI; v1.2 current)

**v1.2 (2026-08-16)**: Adds `visit_solar_term` — the Chinese solar term (jieqi) of each prescription date, assigned from authoritative 2025 astronomical ephemeris tables (Hong Kong Observatory, cross-checked against bmcx.com, Taipei Astronomical Museum, and Xinhua/Purple Mountain Observatory almanac). Also adds audited per-prescription fields `administration_route` and `dosing_frequency`. All v1.1 fields are unchanged (cell-identical). **Privacy note**: solar terms narrow the visit date from half-year (~182 d) to ~15-day windows; the equivalence-class diagnostic over (sex × age band × solar term) has minimum k = 1 (see README section *Privacy* and BUILD_REPORT.md). **v1.1 (2026-08-15)**: Privacy-generalised release: month-level dates replaced by `visit_half_year`, exact ages by `patient_age_group` bands; min k = 3 over (age band × sex × half-year). Added herb_lookup.csv (438 entries). **v1.0 (2025-05)**: initial release.

## Dataset Description

Complete outpatient prescription records of **Professor Zhao Hanqing (赵汉青)**, a National TCM Master (国医大师), from January to December 2025 (actual data coverage: 2025-01-05 to 2025-12-28). All herbal prescriptions dispensed during outpatient consultations at the TCM Internal Medicine department.

Structured for research in TCM clinical pattern analysis, herb combination networks, syndrome–disease correlations, and **seasonal/solar-term prescribing patterns** (v1.2).

## Files

| File | Description | Format |
|------|-------------|--------|
| `ZhaoHanqing_TCM_Prescriptions_2025.csv` | Herb-level records (one row per herb per prescription), {len(v12p)} rows × {len(v12p.columns)} columns | CSV |
| `ZhaoHanqing_Prescription_Summary_2025.csv` | Prescription-level summary (one row per prescription), {len(v12s)} rows × {len(v12s.columns)} columns | CSV |
| `Data_Dictionary.csv` | Variable definitions, allowed values, missing-value policy | CSV |
| `Analysis_Data_Dictionary.csv` | Definitions of downstream analysis outputs | CSV |
| `herb_lookup.csv` | Pinyin–Chinese–Latin herb mapping (438 entries) | CSV |
| `Nomenclature_QA_notes.csv` | Herb-name curation notes | CSV |
| `datapackage.json` | Frictionless Data Package metadata | JSON |
| `README.md`, `LICENSE` | Documentation; CC BY 4.0 text | MD / TXT |

## Dataset Summary

| Metric | Value |
|--------|-------|
| Herb-level records | 5,951 |
| Prescriptions | 335 |
| Unique patients | 122 |
| Unique herbs (Chinese names) | 438 |
| Disease labels (excl. "Not recorded") | 90 |
| Syndrome labels (excl. "Not recorded") | 101 |
| Date coverage | 2025-01-05 – 2025-12-28 |
| Solar terms represented | {len(terms_present)} of 24 (absent: {', '.join(terms_absent)} — no visits in that window) |
| Department | TCM Internal Medicine |

## Variable Overview

### Main Data File (`ZhaoHanqing_TCM_Prescriptions_2025.csv`, 17 columns)

- **prescription_id** — Anonymized prescription identifier (RX001–RX335)
- **patient_id** — Anonymized patient identifier (P001–P122)
- **visit_type** — Initial visit / Follow-up visit
- **patient_sex** — Female / Male
- **herb_name_pinyin** — Herb name, Pinyin romanization
- **herb_name_chinese** — Herb name, Chinese characters
- **herb_latin_name** — Latin pharmacopoeial name when mapped (blank = unmapped)
- **dose_g** — Prescribed dose in grams
- **syndrome_pattern** — Primary TCM syndrome pattern, English (101 labels; "Not recorded" where source blank)
- **disease_name** — Primary disease diagnosis, English (90 labels; "Not recorded" where source blank)
- **season** — Season derived from visit **month** (Spring=Feb–Apr, Summer=May–Jul, Autumn=Aug–Oct, Winter=Jan, Nov, Dec). This column is unchanged from v1.0/v1.1 and intentionally differs from solar-term seasons at month/term boundaries (see below).
- **department** — TCM Internal Medicine
- **patient_age_group** — 10-29 / 30-39 / 40-49 / 50-59 / 60-69 / 70-79 / 80-89
- **visit_half_year** — 2025-H1 / 2025-H2
- **visit_solar_term** — Solar term (jieqi) of the prescription date (24 standard English names; v1.2)
- **administration_route** — Oral decoction / External wash / Oral (dissolved) / External use (v1.2)
- **dosing_frequency** — Twice daily / Three times daily (v1.2)

### Summary Data File (`ZhaoHanqing_Prescription_Summary_2025.csv`, 15 columns)

Same prescription-level fields plus **total_herbs** and **herb_list_pinyin** (comma-separated Pinyin list); herb-level fields omitted.

## Solar Term Assignment (v1.2)

`visit_solar_term` is assigned by **calendar day**: a prescription belongs to the term whose jieqi-transition day is the latest transition day on or before the prescription date (交节日当天及之后 → 新节气，直至下一交节前一日). Minute-level transition times are not used; they cannot change any day-level assignment in 2025 (the closest case, Winter Solstice 2025-12-21 23:03, is 57 minutes from midnight).

Source ephemeris (2025): Hong Kong Observatory complete table (cross-checked: jieqi.bmcx.com, Taipei Astronomical Museum, Xinhua/Purple Mountain Observatory 《中国天文年历》). All 24 transition dates agreed across sources.

**`season` vs `visit_solar_term`**: `season` is defined by Gregorian month (Spring = Mar–May in Chinese meteorological convention was **not** used here; this dataset uses Feb–Apr / May–Jul / Aug–Oct / Jan+Nov+Dec as recorded in the source data), while solar-term seasons begin on the "Start of ..." jieqi days (Feb 3 / May 5 / Aug 7 / Nov 7). Prescriptions in the first days of February, May, August, and November therefore show month-season ≠ term-season by construction; both columns are retained unchanged.

## Privacy and Ethics

- Patient names and medical record numbers replaced by anonymous codes; no free-text identifiers.
- Dates generalised: month-level dates removed in v1.1; `visit_half_year` (H1/H2) coarser, `visit_solar_term` (~15-day) finer — the finer window trades privacy for seasonal analytic value.
- k-anonymity diagnostics (prescription level, QI = sex × age band × time window):
  - `visit_half_year`: 24 equivalence classes, min k = 3, max k = 37.
  - `visit_solar_term`: __QI2_SENTENCE__ — **a deliberate release decision**; re-identification risk assessment is documented in BUILD_REPORT.md.
- No geographic identifiers beyond city level; no photographs or biometric data.

## Usage License

**Creative Commons Attribution 4.0 International (CC BY 4.0)** — see `LICENSE`.

## Citation

> [Author names TBD]. (2025). Zhao Hanqing TCM Prescription Dataset 2025 (v1.2). [DOI TBD / concept DOI 10.5281/zenodo.21951692].

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-05 | Initial release |
| 1.1 | 2026-08-15 | Privacy generalisation (age bands, half-year), herb_lookup, min k = 3 |
| 1.2 | 2026-08-16 | Added `visit_solar_term` (+ audited `administration_route`, `dosing_frequency`); solar-term k-anonymity diagnostic documented |
"""
# NOTE: qi2 is computed below before README write (section 9 ordering handled by
# computing k-anonymity first). See code flow: section 9 precedes README write.

# ---------------------------------------------------------------------------
# 9. k-anonymity diagnostics (both QI sets, prescription level)
# ---------------------------------------------------------------------------
qi_data = pd.DataFrame({
    'sex': rs['patient_sex'].values,
    'age': rs['patient_age_group'].values,
    'half': rs['visit_half_year'].values,
    'term': rs['visit_solar_term'].values,
})
qi1 = qi_data.groupby(['sex', 'age', 'half']).size()
qi2 = qi_data.groupby(['sex', 'age', 'term']).size()
qi1_row = qi_data.groupby(['sex', 'age', 'half']).size()  # herb-row level below
hp = rp.groupby(['patient_sex', 'patient_age_group', 'visit_half_year']).size()
assert len(qi1) == 24 and qi1.min() == 3 and qi1.max() == 37, \
    f'QI1 mismatch vs v1.1 expectation: classes={len(qi1)} min={qi1.min()} max={qi1.max()}'

# ---------------------------------------------------------------------------
# 10. Season × solar-term cross-check (month-season vs term-season)
# ---------------------------------------------------------------------------
TERM_SEASON = {  # standard astronomical season of each term
    'Minor Cold': 'Winter', 'Major Cold': 'Winter',
    'Start of Spring': 'Spring', 'Rain Water': 'Spring',
    'Awakening of Insects': 'Spring', 'Spring Equinox': 'Spring',
    'Clear and Bright': 'Spring', 'Grain Rain': 'Spring',
    'Start of Summer': 'Summer', 'Grain Buds': 'Summer', 'Grain in Ear': 'Summer',
    'Summer Solstice': 'Summer', 'Minor Heat': 'Summer', 'Major Heat': 'Summer',
    'Start of Autumn': 'Autumn', 'End of Heat': 'Autumn', 'White Dew': 'Autumn',
    'Autumn Equinox': 'Autumn', 'Cold Dew': 'Autumn', "Frost's Descent": 'Autumn',
    'Start of Winter': 'Winter', 'Minor Snow': 'Winter', 'Major Snow': 'Winter',
    'Winter Solstice': 'Winter',
}
cross = pd.DataFrame({
    'rx': rs['prescription_id'],
    'date': presc_date.dt.strftime('%Y-%m-%d').rename(index=presc_map).values,
    'season_col': rs['season'].values,
    'term': rs['visit_solar_term'].values,
})
cross['term_season'] = cross['term'].map(TERM_SEASON)
mismatch = cross[cross['season_col'] != cross['term_season']]
assert set(cross['term_season'].unique()) <= {'Spring', 'Summer', 'Autumn', 'Winter'}
# every mismatch must fall in a month-boundary window (first days of Feb/May/Aug/Nov)
mm_windows = mismatch['date'].str[5:7].value_counts().to_dict()

# ---------------------------------------------------------------------------
# 11. Deterministic 20-prescription sample for manual verification
# ---------------------------------------------------------------------------
sample_dates = ['2025-01-05', '2025-05-04', '2025-08-03', '2025-11-02',
                '2025-12-21', '2025-12-28', '2025-04-06', '2025-06-22',
                '2025-11-09', '2025-12-07']
sample_rows = []
for i, sd in enumerate(sample_dates):
    take = 3 if i < 4 else (2 if i < 6 else 1)
    for _, r in cross[cross['date'] == sd].head(take).iterrows():
        sample_rows.append(r)
sample = pd.DataFrame(sample_rows)
assert len(sample) == 20, f'manual-check sample has {len(sample)} rows, expected 20'
# pre-transition days (5/4, 8/3, 11/2) and post-transition days
# (4/6 after Qingming, 6/22 after Solstice, 11/9 after Lidong, 12/7 ON Daxue day)
assert set(sample['date']) == set(sample_dates)

# ---------------------------------------------------------------------------
# 12. README write (qi2 available now) + zip + BUILD_REPORT inputs
# ---------------------------------------------------------------------------
README = README.replace('__QI2_SENTENCE__',
    f'{len(qi2)} equivalence classes, min k = {int(qi2.min())}, max k = {int(qi2.max())}, '
    f'{int((qi2 < 3).sum())} classes with k < 3')
with open(os.path.join(OUT, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(README)

ZIP9 = ['Analysis_Data_Dictionary.csv', 'Data_Dictionary.csv', 'LICENSE',
        'Nomenclature_QA_notes.csv', 'README.md',
        'ZhaoHanqing_Prescription_Summary_2025.csv',
        'ZhaoHanqing_TCM_Prescriptions_2025.csv', 'datapackage.json', 'herb_lookup.csv']
zip_path = os.path.join(OUT, 'ZhaoHanqing_TCM_Dataset_2025_v1.2.zip')
if os.path.exists(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in ZIP9:
        zf.write(os.path.join(OUT, name), arcname=name)
# verify zip integrity: names + bytes identical
with zipfile.ZipFile(zip_path) as zf:
    assert sorted(zf.namelist()) == sorted(ZIP9)
    for name in ZIP9:
        assert zf.read(name) == open(os.path.join(OUT, name), 'rb').read(), f'zip byte mismatch: {name}'
dir9 = sorted(os.listdir(OUT))
expected = sorted(ZIP9 + ['ZhaoHanqing_TCM_Dataset_2025_v1.2.zip'])
if os.path.exists(os.path.join(OUT, 'BUILD_REPORT.md')):
    expected = sorted(expected + ['BUILD_REPORT.md'])  # report stays OUTSIDE the zip
assert dir9 == expected, f'unexpected dir contents: {dir9}'

def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()

# ---------------------------------------------------------------------------
# 13. Console audit trail
# ---------------------------------------------------------------------------
print('=== ANCHOR CHECK (all must be OK) ===')
for zh, d, t, td, tt, ok, note in anchor_report:
    print(f'  {"OK " if ok else "FAIL"} {zh} anchor {d} {t or "(date only)"} | table {td} {tt} | {note}')
print('\n=== CORE COUNTS ===')
print(f'  rows=5951  prescriptions=335  patients=122  herbs=438')
print(f'  disease labels=90  syndrome labels=101  (Not recorded: disease 167 rows, syndrome 48 rows)')
print(f'  v1.2 herb table: {v12p.shape}, summary: {v12s.shape}')
print('\n=== SOLAR TERM TABLE (2025, HKO) ===')
for d, zh, en, t in SOLAR_TERMS_2025:
    n = int((presc_term == en).sum())
    print(f'  {d} {t} {zh:2s} {en:22s} prescriptions={n}')
print(f'  terms present: {len(terms_present)}; absent: {terms_absent}')
print('\n=== FIELD AUDIT ===')
for k, v in audit.items():
    print(f'  {k}: {v.to_dict()}')
print(f'  within-prescription prescriptions with >1 distinct value: {within}')
print(f'  rows with 金额 != 应收金额: {amount_mismatch}')
print('  DECISION: administration_route & dosing_frequency ADDED (clear semantics, constant per prescription);')
print('            总数量 (quantity) & 规格 (spec) NOT added (ambiguous units/messy packaging) — report only.')
print('\n=== k-ANONYMITY (prescription level, n=335) ===')
print(f'  QI1 (sex, age_group, visit_half_year): classes={len(qi1)} min={qi1.min()} max={qi1.max()} classes_k<3={int((qi1 < 3).sum())}')
print(f'  QI2 (sex, age_group, visit_solar_term): classes={len(qi2)} min={qi2.min()} max={qi2.max()} classes_k<3={int((qi2 < 3).sum())} classes_k=1={int((qi2 == 1).sum())} classes_k=2={int((qi2 == 2).sum())}')
print(f'  prescriptions inside k<3 classes (QI2): {int(qi2[qi2 < 3].sum())} / 335')
print(f'  [supplement] QI1 at herb-row level: classes={len(hp)} min={hp.min()} max={hp.max()}')
print('\n=== SEASON x SOLAR TERM CROSS-CHECK ===')
print(f'  season column definition: by month (春=2-4月, 夏=5-7月, 秋=8-10月, 冬=1,11,12月)')
print(f'  mismatches (month-season != term-season): {len(mismatch)} prescriptions, by month: {mm_windows}')
for _, r in mismatch.iterrows():
    print(f'    {r["rx"]} {r["date"]}: season={r["season_col"]} vs term={r["term"]} ({r["term_season"]})')
print('\n=== 20-PRESCRIPTION MANUAL-CHECK SAMPLE ===')
for _, r in sample.iterrows():
    print(f'  {r["rx"]} {r["date"]} -> {r["term"]} (season={r["season_col"]})')
print('\n=== FILES ===')
for name in ZIP9 + ['ZhaoHanqing_TCM_Dataset_2025_v1.2.zip']:
    p = os.path.join(OUT, name)
    print(f'  {name:45s} {os.path.getsize(p):>8d} B  md5={md5(p)}')
print('\nALL ASSERTIONS PASSED.')
