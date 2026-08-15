#!/usr/bin/env python3
"""
赵汉青国医大师门诊方药数据挖掘 - 聚类分析与因子分析
===================================================
1. 处方聚类 (K-Means + 层次聚类 + t-SNE)
2. 药物聚类 (Jaccard + 层次聚类)
3. 因子分析/PCA
4. 患者聚类
5. 处方复杂度分析
"""

import json
import warnings
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── 中文字体 ──
for font in ['WenQuanYi Micro Hei', 'SimHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']:
    try:
        rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy import stats

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / 'figures'
RES = BASE / 'results'
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# ── 加载数据 ──
print("加载数据...")
df = pd.read_pickle(BASE / 'data/cleaned_data.pkl')
print(f"  {len(df)} 条记录, {df['处方号'].nunique()} 张处方, "
      f"{df['病历号　'].nunique()} 位患者, {df['名称'].nunique()} 种药物")

results = {}

# ============================================================
# 辅助：构建处方-药物矩阵
# ============================================================
def build_prescription_drug_matrix(df, use_tfidf=True):
    """构建处方×药物矩阵 (335×438)"""
    pres_col = '处方号'
    drug_col = '名称'
    dose_col = '每次用量'

    # 透视表：行=处方号，列=药物名，值=每次用量之和
    matrix = df.groupby([pres_col, drug_col])[dose_col].sum().unstack(fill_value=0)
    # 确保所有处方都有（包括只有一种药的）
    all_pres = df[pres_col].unique()
    matrix = matrix.reindex(all_pres, fill_value=0).fillna(0)

    if use_tfidf:
        # 二值化后再 TF-IDF
        binary = (matrix > 0).astype(float)
        transformer = TfidfTransformer()
        matrix_tfidf = pd.DataFrame(
            transformer.fit_transform(binary.values).toarray(),
            index=matrix.index,
            columns=matrix.columns
        )
        return matrix_tfidf
    return matrix


# ============================================================
# 1. 处方聚类分析
# ============================================================
print("\n" + "="*60)
print("1. 处方聚类分析")
print("="*60)

pres_matrix = build_prescription_drug_matrix(df, use_tfidf=True)
print(f"  处方-药物矩阵: {pres_matrix.shape}")

# ── 1a. K-Means 肘部法则 ──
print("  K-Means 肘部法则...")
inertias = []
sil_scores = []
K_range = range(2, 16)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(pres_matrix.values)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(pres_matrix.values, labels, sample_size=min(300, len(pres_matrix))))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(list(K_range), inertias, 'bo-', label='Inertia')
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.plot(list(K_range), sil_scores, 'rs-', label='Silhouette')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('K-Means 肘部法则与轮廓系数')
fig.tight_layout()
plt.savefig(FIG / 'prescription_elbow.png', dpi=150, bbox_inches='tight')
plt.close()

# 选最优K（轮廓系数最大）
best_k = list(K_range)[np.argmax(sil_scores)]
print(f"  最优K={best_k} (Silhouette={max(sil_scores):.4f})")

# ── 1b. K-Means 最终聚类 ──
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
pres_labels_km = km_final.fit_predict(pres_matrix.values)

results['prescription_kmeans'] = {
    'optimal_k': int(best_k),
    'silhouette_score': float(max(sil_scores)),
    'cluster_sizes': {str(i): int(np.sum(pres_labels_km == i)) for i in range(best_k)},
    'inertias': {str(k): float(v) for k, v in zip(K_range, inertias)},
    'silhouette_scores': {str(k): float(v) for k, v in zip(K_range, sil_scores)}
}

# 每个聚类的特征药物
print("  各聚类特征药物:")
pres_matrix_raw = build_prescription_drug_matrix(df, use_tfidf=False)
cluster_top_drugs = {}
for c in range(best_k):
    mask = pres_labels_km == c
    cluster_drugs = pres_matrix_raw.iloc[mask].sum().sort_values(ascending=False)
    # 与总体频率比较，找特征药物
    overall_freq = (pres_matrix_raw > 0).mean()
    cluster_freq = (pres_matrix_raw.iloc[mask] > 0).mean()
    lift = (cluster_freq / overall_freq).replace([np.inf, -np.inf], 0).fillna(0)
    top_lift = lift.sort_values(ascending=False).head(15)
    cluster_top_drugs[str(c)] = []
    for drug in top_lift.index:
        cluster_top_drugs[str(c)].append({
            'drug': drug,
            'lift': round(float(top_lift[drug]), 3),
            'freq_in_cluster': round(float(cluster_freq.get(drug, 0)), 3)
        })
    print(f"    Cluster {c} (n={mask.sum()}): {', '.join(top_lift.index[:8].tolist())}")

results['prescription_kmeans']['characteristic_drugs'] = cluster_top_drugs

# 每个聚类的特征证型
print("  各聚类特征证型:")
pres_syndrome = df.groupby('处方号')['主要证型'].first()
cluster_syndromes = {}
for c in range(best_k):
    mask = pres_labels_km == c
    pres_ids = pres_matrix.index[mask]
    syn_counts = pres_syndrome.reindex(pres_ids).value_counts().head(5)
    cluster_syndromes[str(c)] = {s: int(cnt) for s, cnt in syn_counts.items()}
    print(f"    Cluster {c}: {dict(syn_counts)}")

results['prescription_kmeans']['characteristic_syndromes'] = cluster_syndromes

# ── 1c. t-SNE 可视化 ──
print("  t-SNE 降维...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(pres_matrix) - 1))
tsne_coords = tsne.fit_transform(pres_matrix.values)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=pres_labels_km, cmap='tab10', alpha=0.7, s=30)
plt.colorbar(scatter, label='聚类编号')
ax.set_title(f'处方聚类 t-SNE 可视化 (K={best_k})')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.savefig(FIG / 'prescription_tsne.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 1d. 层次聚类 + 树状图 ──
print("  层次聚类树状图...")
# 对处方采样（太多画不下）
n_sample = min(60, len(pres_matrix))
sample_idx = np.random.choice(len(pres_matrix), n_sample, replace=False)
sample_data = pres_matrix.values[sample_idx]
sample_labels_list = [str(pres_matrix.index[i]) for i in sample_idx]

Z_pres = linkage(sample_data, method='ward', metric='euclidean')
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z_pres, labels=sample_labels_list, leaf_rotation=90, ax=ax, leaf_font_size=6)
ax.set_title('处方层次聚类树状图（采样60张）')
ax.set_xlabel('处方号')
ax.set_ylabel('距离')
plt.savefig(FIG / 'prescription_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 2. 药物聚类（系统聚类）
# ============================================================
print("\n" + "="*60)
print("2. 药物聚类（Jaccard + 层次聚类）")
print("="*60)

# 共现矩阵
print("  构建药物共现矩阵...")
pres_binary = (pres_matrix_raw > 0).astype(float).T  # 药物×处方
cooccurrence = pres_binary @ pres_binary.T  # 药物共现次数

# Jaccard 相似度
A = pres_binary.values  # drug × prescription (binary)
intersection = A @ A.T  # 共现次数
row_sums = A.sum(axis=1)
union = row_sums[:, None] + row_sums[None, :] - intersection
jaccard = np.where(union > 0, intersection / union, 0)
jaccard_dist = 1 - jaccard
np.fill_diagonal(jaccard_dist, 0)

# 只聚类出现频次较高的药物（>5次）
drug_freq = pres_binary.sum(axis=1)
top_drugs_mask = drug_freq > 5
top_drug_names = pres_matrix_raw.columns[top_drugs_mask]
print(f"  高频药物(>5次): {top_drug_names.shape[0]} 种")

top_idx = np.where(top_drugs_mask)[0]
Z_drugs = linkage(jaccard_dist[np.ix_(top_idx, top_idx)], method='average')

# 绘制药物聚类树状图（取出现>30次的药物）
freq30_mask = drug_freq > 30
freq30_names = pres_matrix_raw.columns[freq30_mask]
freq30_idx = np.where(freq30_mask)[0]
print(f"  常用药物(>30次): {len(freq30_idx)} 种")

if len(freq30_idx) > 1:
    Z_drugs_freq = linkage(jaccard_dist[np.ix_(freq30_idx, freq30_idx)], method='average')
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(Z_drugs_freq, labels=freq30_names.tolist(), leaf_rotation=90, ax=ax, leaf_font_size=7)
    ax.set_title('药物系统聚类树状图（出现>30次，Jaccard距离）')
    ax.set_xlabel('药物')
    ax.set_ylabel('距离')
    plt.savefig(FIG / 'drug_dendrogram.png', dpi=150, bbox_inches='tight')
    plt.close()

# 发现"药对"——Jaccard相似度最高的药物对
print("  发现药对规律...")
drug_pairs = []
for i in range(len(top_idx)):
    for j in range(i + 1, len(top_idx)):
        ii, jj = top_idx[i], top_idx[j]
        if jaccard[ii, jj] > 0.3:
            drug_pairs.append({
                'drug1': pres_matrix_raw.columns[ii],
                'drug2': pres_matrix_raw.columns[jj],
                'jaccard': round(float(jaccard[ii, jj]), 4),
                'cooccurrence': int(intersection[ii, jj])
            })
drug_pairs.sort(key=lambda x: x['jaccard'], reverse=True)
drug_pairs = drug_pairs[:50]  # top 50
results['drug_pairs'] = drug_pairs
print(f"  发现高相似度药对(>0.3): {len(drug_pairs)} 对")
for p in drug_pairs[:10]:
    print(f"    {p['drug1']} - {p['drug2']}: Jaccard={p['jaccard']}, 共现={p['cooccurrence']}")

# 药物聚类（cut tree at k=8）
drug_clusters_labels = fcluster(Z_drugs, t=8, criterion='maxclust')
drug_cluster_result = {}
for c in range(1, 9):
    mask = drug_clusters_labels == c
    drugs_in_cluster = top_drug_names[mask].tolist()
    if drugs_in_cluster:
        drug_cluster_result[str(c)] = drugs_in_cluster[:20]
        print(f"  药物聚类 {c} ({len(drugs_in_cluster)} 种): {', '.join(drugs_in_cluster[:8])}...")
results['drug_clusters'] = drug_cluster_result


# ============================================================
# 3. 因子分析/PCA
# ============================================================
print("\n" + "="*60)
print("3. 因子分析/PCA")
print("="*60)

# 用处方×药物二值矩阵
pres_binary_mat = (pres_matrix_raw > 0).astype(float)

# PCA
print("  PCA分析...")
pca_full = PCA()
pca_full.fit(pres_binary_mat.values)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_80 = int(np.searchsorted(cumvar, 0.80) + 1)
print(f"  解释80%方差需要 {n_comp_80} 个主成分")
print(f"  前10个主成分解释方差: {cumvar[:10].round(4)}")

# Scree plot
fig, ax = plt.subplots(figsize=(10, 5))
n_show = min(30, len(cumvar))
ax.bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show], alpha=0.6, label='方差解释率')
ax.plot(range(1, n_show + 1), cumvar[:n_show], 'ro-', label='累计方差')
ax.axhline(y=0.8, color='g', linestyle='--', label='80%阈值')
ax.set_xlabel('主成分')
ax.set_ylabel('方差解释率')
ax.set_title('PCA 碎石图')
ax.legend()
plt.savefig(FIG / 'pca_scree.png', dpi=150, bbox_inches='tight')
plt.close()

# 因子分析（取前10个因子）
print("  因子分析 (10因子)...")
n_factors = min(10, pres_binary_mat.shape[1] - 1)
fa = FactorAnalysis(n_components=n_factors, random_state=42)
fa.fit(pres_binary_mat.values)

# 每个因子的Top药物载荷
factor_analysis_result = {}
tcm_methods = []  # 尝试命名因子
for f in range(n_factors):
    loadings = pd.Series(fa.components_[f], index=pres_binary_mat.columns)
    top_pos = loadings.sort_values(ascending=False).head(15)
    top_neg = loadings.sort_values(ascending=False).tail(5)
    factor_analysis_result[f'factor_{f+1}'] = {
        'top_positive_drugs': {drug: round(float(val), 4) for drug, val in top_pos.items()},
        'variance_explained': round(float(pca_full.explained_variance_ratio_[f]), 4) if f < len(pca_full.explained_variance_ratio_) else 0
    }
    print(f"  Factor {f+1}: {', '.join(top_pos.index[:6].tolist())}")

results['factor_analysis'] = factor_analysis_result

# 因子载荷图（前2个因子）
fig, ax = plt.subplots(figsize=(10, 8))
loadings_df = pd.DataFrame(fa.components_[:2].T, index=pres_binary_mat.columns, columns=['F1', 'F2'])
# 只标注载荷较大的
threshold = 0.3
significant = loadings_df[(loadings_df['F1'].abs() > threshold) | (loadings_df['F2'].abs() > threshold)]
ax.scatter(loadings_df['F1'], loadings_df['F2'], alpha=0.3, s=10, c='gray')
ax.scatter(significant['F1'], significant['F2'], alpha=0.7, s=20, c='red')
for idx, row in significant.iterrows():
    ax.annotate(idx, (row['F1'], row['F2']), fontsize=6, alpha=0.8)
ax.set_xlabel('因子 1')
ax.set_ylabel('因子 2')
ax.set_title('因子分析载荷图 (Factor 1 vs Factor 2)')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.savefig(FIG / 'factor_loading.png', dpi=150, bbox_inches='tight')
plt.close()

# PCA 2D散点图
pca_2d = PCA(n_components=2)
coords_2d = pca_2d.fit_transform(pres_binary_mat.values)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=pres_labels_km, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='K-Means聚类')
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
ax.set_title('处方PCA降维散点图（颜色=K-Means聚类）')
plt.savefig(FIG / 'pca_scatter.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 4. 患者聚类
# ============================================================
print("\n" + "="*60)
print("4. 患者聚类")
print("="*60)

print("  构建患者×药物矩阵...")
patient_drug = df.groupby(['病历号　', '名称'])['每次用量'].sum().unstack(fill_value=0)
patient_drug_binary = (patient_drug > 0).astype(float)

# TF-IDF
tfidf = TfidfTransformer()
patient_tfidf = pd.DataFrame(
    tfidf.fit_transform(patient_drug_binary.values).toarray(),
    index=patient_drug.index,
    columns=patient_drug.columns
)

print(f"  患者矩阵: {patient_tfidf.shape}")

# K-Means
patient_k_range = range(2, min(11, len(patient_tfidf)))
p_sil = []
for k in patient_k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lb = km.fit_predict(patient_tfidf.values)
    p_sil.append(silhouette_score(patient_tfidf.values, lb))

best_pk = list(patient_k_range)[np.argmax(p_sil)] if p_sil else 2
print(f"  患者最优K={best_pk} (Silhouette={max(p_sil):.4f})")

km_patient = KMeans(n_clusters=best_pk, random_state=42, n_init=10)
patient_labels = km_patient.fit_predict(patient_tfidf.values)

patient_cluster_info = {}
patient_info = df.groupby('病历号　').agg({
    '性别': 'first', '年龄': 'first', '主要证型': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '',
    '主要疾病': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ''
})

for c in range(best_pk):
    mask = patient_labels == c
    patient_ids = patient_tfidf.index[mask]
    info = patient_info.reindex(patient_ids)
    
    # 特征药物
    drug_usage = patient_drug.iloc[mask].sum().sort_values(ascending=False)
    
    patient_cluster_info[str(c)] = {
        'size': int(mask.sum()),
        'avg_age': round(float(info['年龄'].mean()), 1),
        'gender_dist': info['性别'].value_counts().to_dict(),
        'top_syndromes': info['主要证型'].value_counts().head(5).to_dict(),
        'top_diseases': info['主要疾病'].value_counts().head(5).to_dict(),
        'top_drugs': {drug: int(val) for drug, val in drug_usage.head(10).items()}
    }
    print(f"  患者亚群 {c} (n={mask.sum()}): 平均年龄={info['年龄'].mean():.1f}, "
          f"主要证型={info['主要证型'].value_counts().index[0]}")

results['patient_clusters'] = {
    'optimal_k': int(best_pk),
    'silhouette_score': float(max(p_sil)) if p_sil else 0,
    'clusters': patient_cluster_info
}

# t-SNE 患者可视化
if len(patient_tfidf) >= 5:
    tsne_p = TSNE(n_components=2, random_state=42, perplexity=min(30, len(patient_tfidf) - 1))
    coords_p = tsne_p.fit_transform(patient_tfidf.values)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords_p[:, 0], coords_p[:, 1], c=patient_labels, cmap='tab10', alpha=0.7, s=40)
    plt.colorbar(scatter, label='聚类编号')
    ax.set_title(f'患者聚类 t-SNE 可视化 (K={best_pk})')
    plt.savefig(FIG / 'patient_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 5. 处方复杂度分析
# ============================================================
print("\n" + "="*60)
print("5. 处方复杂度分析")
print("="*60)

# 5a. 处方药味数分布
print("  处方药味数分布...")
drug_count_per_pres = df.groupby('处方号')['名称'].nunique()
print(f"  药味数: 均值={drug_count_per_pres.mean():.1f}, 中位数={drug_count_per_pres.median():.0f}, "
      f"范围=[{drug_count_per_pres.min()}, {drug_count_per_pres.max()}]")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(drug_count_per_pres, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(drug_count_per_pres.mean(), color='r', linestyle='--', label=f'均值={drug_count_per_pres.mean():.1f}')
ax.axvline(drug_count_per_pres.median(), color='g', linestyle='--', label=f'中位数={drug_count_per_pres.median():.0f}')
ax.set_xlabel('药味数')
ax.set_ylabel('处方数')
ax.set_title('处方药味数分布')
ax.legend()
plt.savefig(FIG / 'prescription_drug_count_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# 5b. Shannon entropy（处方多样性）
print("  处方多样性指数...")
pres_entropy = {}
for pid, group in df.groupby('处方号'):
    doses = group['每次用量'].values.astype(float)
    doses = doses[doses > 0]
    if len(doses) > 0:
        p = doses / doses.sum()
        entropy = -np.sum(p * np.log2(p + 1e-10))
        pres_entropy[pid] = entropy

entropy_values = list(pres_entropy.values())
print(f"  Shannon Entropy: 均值={np.mean(entropy_values):.3f}, 标准差={np.std(entropy_values):.3f}")

# 5c. 药物丰富度曲线（物种积累曲线）
print("  药物丰富度曲线...")
pres_order = df.groupby('处方号').first().sort_values('收费日期').index.tolist()
cumulative_drugs = set()
richness = []
for pid in pres_order:
    drugs_in_pres = set(df[df['处方号'] == pid]['名称'].tolist())
    cumulative_drugs.update(drugs_in_pres)
    richness.append(len(cumulative_drugs))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(richness) + 1), richness, 'b-')
ax.set_xlabel('处方数（按时间排序）')
ax.set_ylabel('累计药物种数')
ax.set_title('药物丰富度曲线（物种积累曲线）')
plt.savefig(FIG / 'drug_richness_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 5d. 核心处方 vs 灵活加减
print("  识别核心处方...")
# 找出高频药物组合（核心处方框架）
pres_drug_sets = df.groupby('处方号')['名称'].apply(set)
# 计算每对处方间的Jaccard相似度，找相似处方群
pres_list = list(pres_drug_sets.index)

# 用药味数和独特性来分类
pres_complexity = pd.DataFrame({
    '处方号': drug_count_per_pres.index,
    '药味数': drug_count_per_pres.values,
    'shannon_entropy': [pres_entropy.get(pid, 0) for pid in drug_count_per_pres.index]
})

# 核心药物（出现在>20%处方中的药物）
drug_pres_freq = (pres_matrix_raw > 0).sum() / len(pres_matrix_raw)
core_drugs = drug_pres_freq[drug_pres_freq > 0.20].sort_values(ascending=False)
print(f"  核心药物(>20%处方): {len(core_drugs)} 种")
print(f"  {', '.join(core_drugs.head(10).index.tolist())}")

# 核心处方识别：高Jaccard相似度的处方组
# 计算每个处方与核心药物集的overlap
core_drug_set = set(core_drugs.index)
core_overlap = pres_drug_sets.apply(lambda s: len(s & core_drug_set) / len(core_drug_set) if len(core_drug_set) > 0 else 0)

results['prescription_complexity'] = {
    'drug_count_stats': {
        'mean': round(float(drug_count_per_pres.mean()), 2),
        'median': float(drug_count_per_pres.median()),
        'min': int(drug_count_per_pres.min()),
        'max': int(drug_count_per_pres.max()),
        'std': round(float(drug_count_per_pres.std()), 2)
    },
    'shannon_entropy_stats': {
        'mean': round(float(np.mean(entropy_values)), 4),
        'std': round(float(np.std(entropy_values)), 4),
        'min': round(float(np.min(entropy_values)), 4),
        'max': round(float(np.max(entropy_values)), 4)
    },
    'total_unique_drugs': int(len(cumulative_drugs)),
    'core_drugs_count': int(len(core_drugs)),
    'core_drugs': {drug: round(float(freq), 4) for drug, freq in core_drugs.head(20).items()},
    'richness_final': int(richness[-1]) if richness else 0
}

# 复杂度分布图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(pres_complexity['药味数'], bins=25, edgecolor='black', alpha=0.7)
axes[0].set_title('药味数分布')
axes[0].set_xlabel('药味数')
axes[0].set_ylabel('处方数')

axes[1].hist(pres_complexity['shannon_entropy'], bins=25, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_title('处方多样性指数(Shannon Entropy)分布')
axes[1].set_xlabel('Shannon Entropy')
axes[1].set_ylabel('处方数')

fig.suptitle('处方复杂度分析', fontsize=14)
plt.savefig(FIG / 'prescription_complexity.png', dpi=150, bbox_inches='tight')
plt.close()

# 核心药物使用率条形图
fig, ax = plt.subplots(figsize=(12, 6))
top20_core = core_drugs.head(20)
ax.barh(range(len(top20_core)), top20_core.values, color='steelblue')
ax.set_yticks(range(len(top20_core)))
ax.set_yticklabels(top20_core.index)
ax.set_xlabel('处方使用率')
ax.set_title('核心药物处方使用率 TOP20')
ax.invert_yaxis()
plt.savefig(FIG / 'core_drugs_usage.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 保存结果
# ============================================================
print("\n" + "="*60)
print("保存结果")
print("="*60)

with open(RES / 'cluster_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"  结果已保存到: {RES / 'cluster_results.json'}")
print(f"  图表已保存到: {FIG}/")

# 汇总
print("\n" + "="*60)
print("分析汇总")
print("="*60)
print(f"  ✅ 处方K-Means聚类: K={best_k}, Silhouette={max(sil_scores):.4f}")
print(f"  ✅ 处方层次聚类树状图")
print(f"  ✅ t-SNE 2D可视化")
print(f"  ✅ 药物Jaccard聚类: 发现{len(drug_pairs)}对高相似度药对")
print(f"  ✅ 因子分析: {n_factors}个因子")
print(f"  ✅ PCA: 前{n_comp_80}个主成分解释80%方差")
print(f"  ✅ 患者聚类: K={best_pk}")
print(f"  ✅ 处方复杂度: 平均药味数={drug_count_per_pres.mean():.1f}, 核心药物{len(core_drugs)}种")

figures_list = sorted([f.name for f in FIG.glob('*.png')])
print(f"\n  生成图表 ({len(figures_list)} 张):")
for fig_name in figures_list:
    print(f"    📊 {fig_name}")

print("\n✅ 全部分析完成！")
