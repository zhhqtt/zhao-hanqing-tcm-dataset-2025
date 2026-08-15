#!/usr/bin/env python3
"""
赵汉青国医大师门诊方药数据挖掘 - 关联规则与复杂网络分析
"""

import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import defaultdict
from itertools import combinations

# ── 中文字体 ──────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ── 路径 ──────────────────────────────────────────────────
BASE = '/home/zhhq/.openclaw/workspace-coder/ZhaoAnalysis'
FIG  = f'{BASE}/figures'
RES  = f'{BASE}/results'

import os; os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────
df = pd.read_pickle(f'{BASE}/data/cleaned_data.pkl')
print(f"数据: {df.shape[0]} 行, {df['处方号'].nunique()} 张处方, {df['名称'].nunique()} 种药物")

# ============================================================
# 1. 药-药关联规则挖掘 (Apriori)
# ============================================================
print("\n=== 1. 药-药关联规则 ===")

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 构建事务矩阵：行=处方号, 列=药物, 值=是否出现
basket = df.groupby(['处方号', '名称'])['总数量'].sum().unstack().fillna(0)
basket = (basket > 0).astype(int)
print(f"事务矩阵: {basket.shape[0]} 处方 × {basket.shape[1]} 药物")

# Apriori
freq_items = apriori(basket, min_support=0.05, use_colnames=True, max_len=3)
print(f"频繁项集数: {len(freq_items)}")

rules = association_rules(freq_items, metric='confidence', min_threshold=0.5)
rules['ant_len'] = rules['antecedents'].apply(len)
rules['con_len'] = rules['consequents'].apply(len)
print(f"关联规则数: {len(rules)}")

# 提取2-itemset药对和3-itemset药组
pair_rules = rules[(rules['ant_len'] == 1) & (rules['con_len'] == 1)].copy()
triple_rules = rules[(rules['ant_len'] + rules['con_len']) >= 3].copy()

# Top30药对
top_pairs = pair_rules.sort_values('support', ascending=False).head(30)
print(f"\nTop10药对:")
for _, r in top_pairs.head(10).iterrows():
    a, c = list(r['antecedents'])[0], list(r['consequents'])[0]
    print(f"  {a} → {c}: 支持度={r['support']:.4f}, 置信度={r['confidence']:.4f}, 提升度={r['lift']:.4f}")

# Top30药对网络图
fig, ax = plt.subplots(figsize=(14, 12))
G_pair = nx.Graph()
for _, r in top_pairs.iterrows():
    a, c = list(r['antecedents'])[0], list(r['consequents'])[0]
    G_pair.add_edge(a, c, weight=r['support'], confidence=r['confidence'], lift=r['lift'])

pos = nx.spring_layout(G_pair, k=2, seed=42)
edges = G_pair.edges(data=True)
weights = [d['weight'] * 30 for _, _, d in edges]
nx.draw_networkx_edges(G_pair, pos, width=weights, alpha=0.6, edge_color='steelblue', ax=ax)
nx.draw_networkx_nodes(G_pair, pos, node_size=600, node_color='lightcoral', alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_pair, pos, font_size=10, font_family='Noto Sans CJK JP', ax=ax)
ax.set_title('Top30药对关联网络图（边粗细=支持度）', fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{FIG}/01_top30_pair_network.png', dpi=150, bbox_inches='tight')
plt.close()
print("图已保存: 01_top30_pair_network.png")

# ============================================================
# 2. 药-证关联分析 (PMI)
# ============================================================
print("\n=== 2. 药-证关联 ===")

# 展开证型列表
df_syndromes = df.explode('证型列表').drop_duplicates(['处方号', '名称', '证型列表']).reset_index(drop=True)
herd_syndrome = pd.crosstab(df_syndromes['名称'], df_syndromes['证型列表'])

# PMI计算
def pmi_matrix(ct):
    total = ct.values.sum()
    p_xy = ct / total
    p_x = ct.sum(axis=1).values.reshape(-1, 1) / total
    p_y = ct.sum(axis=0).values.reshape(1, -1) / total
    pmi = np.log2(p_xy / (p_x * p_y + 1e-10) + 1e-10)
    pmi = pmi.replace([np.inf, -np.inf], 0).fillna(0)
    return pmi

pmi_hs = pmi_matrix(herd_syndrome)

# Top20药物 × Top15证型
top20_herbs = herd_syndrome.sum(axis=1).nlargest(20).index
top15_syndromes = herd_syndrome.sum(axis=0).nlargest(15).index
pmi_sub = pmi_hs.loc[pmi_hs.index.isin(top20_herbs), pmi_hs.columns.isin(top15_syndromes)]
# Reorder
pmi_sub = pmi_sub.reindex(index=top20_herbs, columns=top15_syndromes).fillna(0)

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(pmi_sub.values, cmap='RdBu_r', aspect='auto')
ax.set_xticks(range(len(top15_syndromes)))
ax.set_xticklabels(top15_syndromes, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(top20_herbs)))
ax.set_yticklabels(top20_herbs, fontsize=9)
plt.colorbar(im, ax=ax, label='PMI')
ax.set_title('Top20药物×Top15证型 PMI热力图', fontsize=16)
plt.tight_layout()
plt.savefig(f'{FIG}/02_herb_syndrome_pmi_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("图已保存: 02_herb_syndrome_pmi_heatmap.png")

# ============================================================
# 3. 药-病关联分析
# ============================================================
print("\n=== 3. 药-病关联 ===")

df_diseases = df.explode('疾病列表').drop_duplicates(['处方号', '名称', '疾病列表']).reset_index(drop=True)
herb_disease = pd.crosstab(df_diseases['名称'], df_diseases['疾病列表'])

# Jaccard-like: 共现频率标准化
hd_norm = herb_disease.div(herb_disease.sum(axis=1), axis=0)

top20_herbs_d = herb_disease.sum(axis=1).nlargest(20).index
top15_diseases = herb_disease.sum(axis=0).nlargest(15).index
hd_sub = hd_norm.reindex(index=top20_herbs_d, columns=top15_diseases).fillna(0)

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(hd_sub.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(top15_diseases)))
ax.set_xticklabels(top15_diseases, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(top20_herbs_d)))
ax.set_yticklabels(top20_herbs_d, fontsize=9)
plt.colorbar(im, ax=ax, label='条件频率')
ax.set_title('Top20药物×Top15疾病 关联热力图', fontsize=16)
plt.tight_layout()
plt.savefig(f'{FIG}/03_herb_disease_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("图已保存: 03_herb_disease_heatmap.png")

# ============================================================
# 4. 复杂网络分析
# ============================================================
print("\n=== 4. 复杂网络分析 ===")

# 构建共现网络
prescription_herbs = df.groupby('处方号')['名称'].apply(set)
G = nx.Graph()
for herbs in prescription_herbs:
    for a, b in combinations(herbs, 2):
        if G.has_edge(a, b):
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a, b, weight=1)

print(f"共现网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# 过滤低频边（出现>=3次）
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 3]
G.remove_edges_from(edges_to_remove)
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
print(f"过滤后: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# 网络指标
degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G)
closeness_cent = nx.closeness_centrality(G)

# 核心药物 Top20
core_herbs = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:20]
print("\nTop10核心药物(度中心性):")
for herb, val in core_herbs[:10]:
    print(f"  {herb}: 度中心性={val:.4f}, 介数={betweenness_cent[herb]:.4f}")

# 社区检测
import community as community_louvain
partition = community_louvain.best_partition(G, weight='weight')
modularity = community_louvain.modularity(partition, G, weight='weight')
n_communities = len(set(partition.values()))
print(f"\n社区检测: {n_communities} 个社区, 模块度={modularity:.4f}")

# 各社区核心药物
communities = defaultdict(list)
for node, comm in partition.items():
    communities[comm].append(node)

print("\n各社区Top5药物:")
for comm_id in sorted(communities.keys())[:min(10, n_communities)]:
    members = communities[comm_id]
    top_members = sorted(members, key=lambda x: degree_cent[x], reverse=True)[:5]
    print(f"  社区{comm_id} ({len(members)}药): {', '.join(top_members)}")

# 网络可视化
fig, ax = plt.subplots(figsize=(20, 18))
pos = nx.spring_layout(G, k=1.5, seed=42, weight='weight')

# 节点大小=度中心性, 颜色=社区
node_sizes = [degree_cent[n] * 5000 + 100 for n in G.nodes()]
comm_colors = [partition[n] for n in G.nodes()]

edges = G.edges(data=True)
edge_widths = [d['weight'] / 5 for _, _, d in edges]

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray', ax=ax)
nc = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                             node_color=comm_colors, cmap=plt.cm.tab20, alpha=0.85, ax=ax)

# 只标注度中心性Top30的节点
top_labels = dict(sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:30])
labels = {n: n for n in G.nodes() if n in top_labels}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='Noto Sans CJK JP', ax=ax)

ax.set_title(f'药物共现复杂网络（{n_communities}个社区，模块度={modularity:.3f}）', fontsize=18)
ax.axis('off')
plt.colorbar(nc, ax=ax, label='社区编号', shrink=0.6)
plt.tight_layout()
plt.savefig(f'{FIG}/04_cooccurrence_network.png', dpi=150, bbox_inches='tight')
plt.close()
print("图已保存: 04_cooccurrence_network.png")

# ============================================================
# 5. 证-病关联
# ============================================================
print("\n=== 5. 证-病关联 ===")

syn_dis_data = df[['处方号', '证型列表', '疾病列表']].drop_duplicates('处方号')
syn_rows = []
for _, row in syn_dis_data.iterrows():
    for s in row['证型列表']:
        for d in row['疾病列表']:
            syn_rows.append({'证型': s, '疾病': d})
syn_dis_df = pd.DataFrame(syn_rows)
syn_dis_ct = pd.crosstab(syn_dis_df['证型'], syn_dis_df['疾病'])

top15_syn = syn_dis_ct.sum(axis=1).nlargest(15).index
top15_dis = syn_dis_ct.sum(axis=0).nlargest(15).index
syn_dis_sub = syn_dis_ct.reindex(index=top15_syn, columns=top15_dis).fillna(0)

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(syn_dis_sub.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(top15_dis)))
ax.set_xticklabels(top15_dis, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(top15_syn)))
ax.set_yticklabels(top15_syn, fontsize=9)
plt.colorbar(im, ax=ax, label='共现次数')
ax.set_title('Top15证型×Top15疾病 共现热力图', fontsize=16)
plt.tight_layout()
plt.savefig(f'{FIG}/05_syndrome_disease_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("图已保存: 05_syndrome_disease_heatmap.png")

# ============================================================
# 保存结果
# ============================================================
print("\n=== 保存结果 ===")

results = {
    "频繁项集数量": len(freq_items),
    "关联规则数量": len(rules),
    "药对规则数量": len(pair_rules),
    "药组规则数量": len(triple_rules),
    "top10药对": [
        {
            "前件": list(r['antecedents'])[0],
            "后件": list(r['consequents'])[0],
            "支持度": round(r['support'], 4),
            "置信度": round(r['confidence'], 4),
            "提升度": round(r['lift'], 4)
        }
        for _, r in top_pairs.head(10).iterrows()
    ],
    "网络节点数": G.number_of_nodes(),
    "网络边数": G.number_of_edges(),
    "社区数": n_communities,
    "模块度": round(modularity, 4),
    "top10核心药物": [
        {"药物": h, "度中心性": round(degree_cent[h], 4), "介数中心性": round(betweenness_cent[h], 4), "接近中心性": round(closeness_cent[h], 4)}
        for h, _ in core_herbs[:10]
    ],
    "社区组成": {
        str(cid): sorted(members, key=lambda x: degree_cent[x], reverse=True)[:5]
        for cid, members in sorted(communities.items())[:min(10, n_communities)]
    }
}

with open(f'{RES}/association_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已保存: {RES}/association_results.json")
print("\n✅ 全部分析完成！")
