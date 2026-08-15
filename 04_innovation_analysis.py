#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
赵汉青国医大师门诊方药数据挖掘 - 创新分析方法
04_innovation_analysis.py
"""

import json
import warnings
import itertools
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')

# ============================================================
# 字体设置
# ============================================================
import matplotlib.font_manager as fm
_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_fm_entry = fm.FontEntry(fname=_font_path, name='NotoSansCJK')
fm.fontManager.ttflist.append(_fm_entry)
FONT_NAME = 'NotoSansCJK'
plt.rcParams['font.sans-serif'] = [FONT_NAME, 'Noto Sans CJK SC', 'SimHei', 'WQY Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

FIG_DIR = 'figures'
RES_FILE = 'results/innovation_results.json'

# ============================================================
# 中药功效分类字典（常见药物）
# ============================================================
HERB_EFFICACY = {
    # 补气药
    '黄芪': '补气', '党参': '补气', '白术': '补气健脾', '炙甘草': '补气', '甘草': '补气',
    '山药': '补气健脾', '太子参': '补气', '人参': '大补元气',
    # 补血药
    '当归': '补血活血', '白芍': '补血', '熟地黄': '补血', '阿胶': '补血',
    '何首乌': '补血', '龙眼肉': '补血',
    # 补阴药
    '麦冬': '补阴', '北沙参': '补阴', '南沙参': '补阴', '百合': '补阴',
    '石斛': '补阴', '枸杞子': '补阴', '女贞子': '补阴', '墨旱莲': '补阴',
    '龟甲': '补阴', '鳖甲': '补阴', '天冬': '补阴', '玉竹': '补阴',
    # 补阳药
    '杜仲': '补阳', '续断': '补阳', '菟丝子': '补阳', '淫羊藿': '补阳',
    '巴戟天': '补阳', '肉苁蓉': '补阳', '补骨脂': '补阳', '鹿角胶': '补阳',
    '仙茅': '补阳', '益智仁': '补阳', '锁阳': '补阳',
    # 活血化瘀药
    '丹参': '活血化瘀', '川芎': '活血化瘀', '赤芍': '活血化瘀', '红花': '活血化瘀',
    '桃仁': '活血化瘀', '牛膝': '活血化瘀', '鸡血藤': '活血化瘀', '郁金': '活血化瘀',
    '延胡索': '活血化瘀', '乳香': '活血化瘀', '没药': '活血化瘀', '三棱': '活血化瘀',
    '莪术': '活血化瘀', '水蛭': '活血化瘀', '地龙': '活血化瘀', '王不留行': '活血化瘀',
    '益母草': '活血化瘀', '泽兰': '活血化瘀', '五灵脂': '活血化瘀', '蒲黄': '活血化瘀',
    '姜黄': '活血化瘀', '降香': '活血化瘀', '血竭': '活血化瘀',
    # 理气药
    '陈皮': '理气', '枳实': '理气', '枳壳': '理气', '香附': '理气', '木香': '理气',
    '乌药': '理气', '厚朴': '理气', '砂仁': '理气', '佛手': '理气', '香橼': '理气',
    '青皮': '理气', '薤白': '理气', '大腹皮': '理气', '沉香': '理气',
    # 疏肝理气
    '柴胡': '疏肝解郁', '郁金': '疏肝解郁',
    # 清热药
    '黄芩': '清热', '黄连': '清热', '黄柏': '清热', '栀子': '清热', '龙胆': '清热',
    '金银花': '清热解毒', '连翘': '清热解毒', '蒲公英': '清热解毒', '紫花地丁': '清热解毒',
    '板蓝根': '清热解毒', '大青叶': '清热解毒', '鱼腥草': '清热解毒',
    '石膏': '清热泻火', '知母': '清热泻火', '天花粉': '清热泻火', '芦根': '清热泻火',
    '夏枯草': '清肝泻火', '决明子': '清肝泻火', '青葙子': '清肝泻火',
    '生地黄': '清热凉血', '玄参': '清热凉血', '牡丹皮': '清热凉血', '紫草': '清热凉血',
    '地骨皮': '清虚热', '银柴胡': '清虚热', '胡黄连': '清虚热', '白薇': '清虚热',
    '白花蛇舌草': '清热解毒', '半枝莲': '清热解毒',
    # 化痰止咳药
    '半夏': '燥湿化痰', '天南星': '燥湿化痰', '白芥子': '燥湿化痰',
    '浙贝母': '清热化痰', '川贝母': '清热化痰', '瓜蒌': '清热化痰', '竹茹': '清热化痰',
    '桔梗': '化痰止咳', '杏仁': '化痰止咳', '紫菀': '化痰止咳', '款冬花': '化痰止咳',
    '百部': '化痰止咳', '桑白皮': '化痰止咳', '葶苈子': '化痰止咳',
    '旋覆花': '化痰', '白前': '化痰', '前胡': '化痰',
    # 祛湿药
    '茯苓': '利水渗湿', '泽泻': '利水渗湿', '薏苡仁': '利水渗湿', '猪苓': '利水渗湿',
    '车前子': '利水渗湿', '滑石': '利水渗湿', '木通': '利水渗湿', '通草': '利水渗湿',
    '瞿麦': '利水渗湿', '萹蓄': '利水渗湿',
    # 祛风湿药
    '独活': '祛风湿', '威灵仙': '祛风湿', '秦艽': '祛风湿', '防己': '祛风湿',
    '桑枝': '祛风湿', '络石藤': '祛风湿', '海风藤': '祛风湿', '木瓜': '祛风湿',
    '伸筋草': '祛风湿', '透骨草': '祛风湿', '雷公藤': '祛风湿',
    # 温里药
    '干姜': '温中散寒', '附子': '温中散寒', '肉桂': '温中散寒', '吴茱萸': '温中散寒',
    '小茴香': '温中散寒', '丁香': '温中散寒', '高良姜': '温中散寒', '花椒': '温中散寒',
    '胡椒': '温中散寒',
    # 解表药
    '桂枝': '解表', '麻黄': '解表', '荆芥': '解表', '防风': '解表', '羌活': '解表',
    '白芷': '解表', '细辛': '解表', '藁本': '解表', '苍耳子': '解表', '辛夷': '解表',
    '薄荷': '解表', '牛蒡子': '解表', '蝉蜕': '解表', '桑叶': '解表', '菊花': '解表',
    '葛根': '解表', '柴胡': '解表', '升麻': '解表',
    # 安神药
    '酸枣仁': '安神', '柏子仁': '安神', '远志': '安神', '合欢皮': '安神',
    '首乌藤': '安神', '茯神': '安神', '龙骨': '安神', '牡蛎': '安神',
    '珍珠母': '安神', '磁石': '安神', '朱砂': '安神', '琥珀': '安神',
    '紫石英': '安神', '煅紫石英': '安神',
    # 平肝息风药
    '天麻': '平肝息风', '钩藤': '平肝息风', '石决明': '平肝息风', '代赭石': '平肝息风',
    '全蝎': '平肝息风', '蜈蚣': '平肝息风', '僵蚕': '平肝息风', '地龙': '平肝息风',
    '珍珠母': '平肝息风', '羚羊角': '平肝息风',
    # 收涩药
    '五味子': '收涩', '乌梅': '收涩', '山茱萸': '收涩', '金樱子': '收涩',
    '芡实': '收涩', '覆盆子': '收涩', '桑螵蛸': '收涩', '莲子': '收涩',
    '浮小麦': '收涩', '麻黄根': '收涩', '椿皮': '收涩',
    # 消食药
    '山楂': '消食', '神曲': '消食', '麦芽': '消食', '谷芽': '消食', '莱菔子': '消食',
    '鸡内金': '消食',
    # 泻下药
    '大黄': '泻下', '芒硝': '泻下', '番泻叶': '泻下', '火麻仁': '泻下',
    '郁李仁': '泻下',
    # 止血药
    '三七': '止血', '蒲黄': '止血', '白及': '止血', '仙鹤草': '止血',
    '地榆': '止血', '槐花': '止血', '侧柏叶': '止血', '白茅根': '止血',
    '藕节': '止血', '棕榈炭': '止血', '血余炭': '止血',
    # 开窍药
    '麝香': '开窍', '冰片': '开窍', '苏合香': '开窍', '石菖蒲': '开窍',
    # 利水渗湿/化石
    '海金沙': '利水渗湿', '金钱草': '利水渗湿',
    # 其他
    '皂角刺': '消肿排脓', '穿山甲': '通络', '炙黄芪': '补气', '麸炒白术': '补气健脾',
    '炒白术': '补气健脾', '焦白术': '补气健脾', '土白术': '补气健脾',
    '焦山楂': '消食', '炒山楂': '消食', '生山楂': '消食',
    '焦神曲': '消食', '炒麦芽': '消食',
    '法半夏': '燥湿化痰', '清半夏': '燥湿化痰', '姜半夏': '燥湿化痰',
    '炙麻黄': '解表', '蜜麻黄': '解表',
    '醋延胡索': '活血化瘀', '醋柴胡': '疏肝解郁',
    '炙枇杷叶': '化痰止咳', '蜜百部': '化痰止咳',
    '姜竹茹': '清热化痰', '姜半夏': '燥湿化痰',
    '炒酸枣仁': '安神', '制远志': '安神',
    '酒白芍': '补血', '酒当归': '补血活血',
    '酒川芎': '活血化瘀', '醋香附': '理气',
    '盐杜仲': '补阳', '盐补骨脂': '补阳',
    '制何首乌': '补血', '制巴戟天': '补阳',
    '烫骨碎补': '补阳', '炒苍耳子': '解表',
    '蜜款冬花': '化痰止咳', '蜜紫菀': '化痰止咳',
    '盐泽泻': '利水渗湿', '盐知母': '清热泻火',
    '麸炒枳壳': '理气', '麸炒苍术': '燥湿健脾',
    '焦栀子': '清热', '炒栀子': '清热',
    '燀苦杏仁': '化痰止咳', '燀桃仁': '活血化瘀',
    '酒丹参': '活血化瘀', '醋乳香': '活血化瘀',
    '醋没药': '活血化瘀', '酒牛膝': '活血化瘀',
    '盐车前子': '利水渗湿', '盐菟丝子': '补阳',
    '蜜百合': '补阴', '蜜前胡': '化痰',
    '胆南星': '清热化痰', '制天南星': '燥湿化痰',
    '炒蒺藜': '平肝息风', '刺蒺藜': '平肝息风',
}

# 治法映射
EFFICACY_TO_METHOD = {
    '补气': '补气法', '补气健脾': '健脾益气法', '大补元气': '补气法',
    '补血': '补血法', '补血活血': '补血活血法',
    '补阴': '滋阴法', '补阳': '温阳法',
    '活血化瘀': '活血化瘀法', '疏肝解郁': '疏肝解郁法',
    '理气': '理气法', '清热': '清热法', '清热解毒': '清热解毒法',
    '清热泻火': '清热泻火法', '清热凉血': '清热凉血法', '清肝泻火': '清肝泻火法',
    '清虚热': '清虚热法', '燥湿化痰': '燥湿化痰法', '清热化痰': '清热化痰法',
    '化痰止咳': '止咳化痰法', '化痰': '化痰法',
    '利水渗湿': '利水渗湿法', '祛风湿': '祛风湿法',
    '温中散寒': '温中散寒法', '解表': '解表法',
    '安神': '安神法', '平肝息风': '平肝息风法',
    '收涩': '固涩法', '消食': '消食法', '泻下': '泻下法',
    '止血': '止血法', '开窍': '开窍法',
    '消肿排脓': '消肿排脓法', '通络': '通络法', '燥湿健脾': '燥湿健脾法',
}

# 经典方剂组成（核心药物）
CLASSIC_FORMULAS = {
    '桂枝汤': {'桂枝', '白芍', '甘草', '生姜', '大枣'},
    '逍遥散': {'柴胡', '当归', '白芍', '白术', '茯苓', '甘草', '薄荷', '生姜'},
    '六味地黄丸': {'熟地黄', '山药', '山茱萸', '茯苓', '牡丹皮', '泽泻'},
    '四君子汤': {'人参', '白术', '茯苓', '甘草'},
    '四物汤': {'当归', '川芎', '白芍', '熟地黄'},
    '血府逐瘀汤': {'桃仁', '红花', '当归', '生地黄', '川芎', '赤芍', '牛膝', '桔梗', '柴胡', '枳壳', '甘草'},
    '归脾汤': {'黄芪', '人参', '白术', '当归', '甘草', '茯神', '远志', '酸枣仁', '木香', '龙眼肉', '生姜', '大枣'},
    '小柴胡汤': {'柴胡', '黄芩', '半夏', '人参', '甘草', '生姜', '大枣'},
    '天麻钩藤饮': {'天麻', '钩藤', '石决明', '栀子', '黄芩', '牛膝', '杜仲', '益母草', '桑寄生', '夜交藤', '茯神'},
    '酸枣仁汤': {'酸枣仁', '甘草', '知母', '茯苓', '川芎'},
    '二陈汤': {'半夏', '陈皮', '茯苓', '甘草'},
    '逍遥散加减': {'柴胡', '当归', '白芍', '白术', '茯苓', '甘草'},
}

print("=" * 60)
print("赵汉青国医大师门诊方药数据挖掘 - 创新分析")
print("=" * 60)

# ============================================================
# 加载数据
# ============================================================
print("\n[加载] 读取清洗后数据...")
df = pd.read_pickle('data/cleaned_data.pkl')
print(f"  记录数: {len(df)}, 处方数: {df['处方号'].nunique()}, "
      f"患者数: {df['姓名'].nunique()}, 药物数: {df['名称'].nunique()}")

results = {}

# ============================================================
# 辅助函数
# ============================================================
def get_prescription_drugs(df, prescription_id):
    """获取一张处方的所有药物集合"""
    drugs = df[df['处方号'] == prescription_id]['名称'].unique()
    return set(drugs)

def jaccard_similarity(set1, set2):
    """Jaccard相似度"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def get_herb_efficacy(herb_name):
    """获取药物功效"""
    return HERB_EFFICACY.get(herb_name, '其他')

def get_treatment_method(efficacy):
    """从功效映射到治法"""
    return EFFICACY_TO_METHOD.get(efficacy, '其他治法')


# ============================================================
# 1. 处方演化路径分析
# ============================================================
print("\n" + "=" * 60)
print("1. 处方演化路径分析")
print("=" * 60)

# 获取每位患者的处方时间序列
patient_visits = df.groupby('姓名').agg(
    prescriptions=('处方号', lambda x: list(x.unique())),
    dates=('收费日期', lambda x: sorted(x.unique()))
).reset_index()

# 筛选多次就诊患者
multi_visit = patient_visits[patient_visits['prescriptions'].apply(len) >= 3].copy()
print(f"  就诊≥3次的患者: {len(multi_visit)}人")

# 计算处方间Jaccard相似度
evolution_data = []
for _, row in multi_visit.iterrows():
    patient = row['姓名']
    prescs = row['prescriptions']
    if len(prescs) < 2:
        continue
    drug_sets = [get_prescription_drugs(df, p) for p in prescs]
    for i in range(len(drug_sets) - 1):
        sim = jaccard_similarity(drug_sets[i], drug_sets[i + 1])
        evolution_data.append({
            'patient': patient,
            'visit_pair': f"V{i+1}-V{i+2}",
            'similarity': sim,
            'pair_index': i
        })

evo_df = pd.DataFrame(evolution_data)

# 基础方识别：所有处方中共同出现的药物
base_herb_results = {}
for _, row in multi_visit.iterrows():
    patient = row['姓名']
    prescs = row['prescriptions']
    drug_sets = [get_prescription_drugs(df, p) for p in prescs]
    if len(drug_sets) < 2:
        continue
    common_drugs = set.intersection(*drug_sets) if drug_sets else set()
    all_drugs = set.union(*drug_sets) if drug_sets else set()
    base_herb_results[patient] = {
        'visit_count': len(prescs),
        'common_drugs': sorted(list(common_drugs)),
        'common_count': len(common_drugs),
        'total_unique_drugs': len(all_drugs),
        'base_ratio': len(common_drugs) / len(all_drugs) if all_drugs else 0
    }

# 选出代表性患者（就诊次数最多且基础方明显）
representative_patients = sorted(base_herb_results.items(),
                                  key=lambda x: (-x[1]['common_count'], -x[1]['visit_count']))[:5]

print(f"  代表性患者基础方:")
for patient, info in representative_patients:
    print(f"    {patient}: 就诊{info['visit_count']}次, "
          f"基础药{info['common_count']}种 ({', '.join(info['common_drugs'][:8])}...)")

# 绘制处方演化相似度曲线（代表性患者）
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：Jaccard相似度整体分布
if not evo_df.empty:
    axes[0].hist(evo_df['similarity'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(evo_df['similarity'].mean(), color='red', linestyle='--', label=f'均值={evo_df["similarity"].mean():.3f}')
    axes[0].set_xlabel('Jaccard相似度', fontsize=12)
    axes[0].set_ylabel('频次', fontsize=12)
    axes[0].set_title('相邻处方间Jaccard相似度分布', fontsize=14)
    axes[0].legend(fontsize=11)

# 右图：代表性患者的演化曲线
colors = plt.cm.Set2(np.linspace(0, 1, len(representative_patients)))
for idx, (patient, info) in enumerate(representative_patients[:4]):
    patient_evo = evo_df[evo_df['patient'] == patient]
    if not patient_evo.empty:
        axes[1].plot(range(len(patient_evo)), patient_evo['similarity'].values,
                     marker='o', label=f'{patient}({info["visit_count"]}次)',
                     color=colors[idx], linewidth=2, markersize=6)

axes[1].set_xlabel('就诊序次转换', fontsize=12)
axes[1].set_ylabel('Jaccard相似度', fontsize=12)
axes[1].set_title('代表性患者处方演化曲线', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_prescription_evolution.png')
plt.close()
print("  [图] 01_prescription_evolution.png 已保存")

# 桑基图数据（代表性患者1的处方演化）
if len(representative_patients) > 0:
    rep_patient = representative_patients[0][0]
    rep_prescs = multi_visit[multi_visit['姓名'] == rep_patient]['prescriptions'].values[0]
    drug_sets = [get_prescription_drugs(df, p) for p in rep_prescs]

    # 构建演化网络图
    G_evo = nx.DiGraph()
    for i, (presc, drugs) in enumerate(zip(rep_prescs, drug_sets)):
        node_label = f"V{i+1}\n({', '.join(list(drugs)[:3])}...)"
        G_evo.add_node(node_label, layer=i, drugs=drugs)
        if i > 0:
            prev_drugs = drug_sets[i - 1]
            kept = drugs & prev_drugs
            removed = prev_drugs - drugs
            added = drugs - prev_drugs
            G_evo.add_edge(
                f"V{i}\n({', '.join(list(drug_sets[i-1])[:3])}...)",
                node_label,
                kept=len(kept), removed=len(removed), added=len(added)
            )

    fig, ax = plt.subplots(figsize=(16, 8))
    pos = {}
    for i, (presc, drugs) in enumerate(zip(rep_prescs, drug_sets)):
        label = f"第{i+1}诊\n{len(drugs)}味药"
        pos[label] = (i * 3, 0)
        node_size = len(drugs) * 100
        ax.scatter(i * 3, 0, s=node_size, c='steelblue', alpha=0.7, zorder=5)
        ax.annotate(label, (i * 3, 0), textcoords="offset points",
                   xytext=(0, 30), ha='center', fontsize=10, fontweight='bold')

        # 显示药物列表
        drug_text = ', '.join(sorted(drugs)[:10])
        if len(drugs) > 10:
            drug_text += f'...共{len(drugs)}味'
        ax.annotate(drug_text, (i * 3, 0), textcoords="offset points",
                   xytext=(0, -30), ha='center', fontsize=7, color='gray')

    # 绘制连线
    for i in range(1, len(drug_sets)):
        kept = len(drug_sets[i] & drug_sets[i - 1])
        total = len(drug_sets[i - 1])
        sim = kept / total if total > 0 else 0
        ax.annotate('', xy=((i) * 3, 0), xytext=((i - 1) * 3, 0),
                   arrowprops=dict(arrowstyle='->', lw=sim * 5 + 1, color=plt.cm.RdYlGn(sim)))
        ax.text((i - 0.5) * 3, 0.3, f'保留{kept}味\n相似度{sim:.0%}',
               ha='center', fontsize=8, color='darkred')

    ax.set_xlim(-1, len(drug_sets) * 3 - 1)
    ax.set_ylim(-0.8, 0.8)
    ax.axis('off')
    ax.set_title(f'患者"{rep_patient}"处方演化路径（共{len(rep_prescs)}次就诊）',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/01b_evolution_sankey.png')
    plt.close()
    print("  [图] 01b_evolution_sankey.png 已保存")

results['prescription_evolution'] = {
    'multi_visit_patients': len(multi_visit),
    'mean_jaccard': float(evo_df['similarity'].mean()) if not evo_df.empty else 0,
    'representative_patients': [
        {'name': p, **info} for p, info in representative_patients[:5]
    ]
}


# ============================================================
# 2. 中药知识图谱构建
# ============================================================
print("\n" + "=" * 60)
print("2. 中药知识图谱构建")
print("=" * 60)

# 构建药物-功效-证型-疾病关联
herb_syndrome = defaultdict(lambda: defaultdict(int))  # herb -> syndrome -> count
herb_disease = defaultdict(lambda: defaultdict(int))    # herb -> disease -> count
syndrome_disease = defaultdict(lambda: defaultdict(int))

for _, row in df.iterrows():
    herb = row['名称']
    syndrome = row['主要证型']
    disease = row['主要疾病']
    efficacy = get_herb_efficacy(herb)

    herb_syndrome[herb][syndrome] += 1
    herb_disease[herb][disease] += 1
    syndrome_disease[syndrome][disease] += 1

# 构建知识图谱
G_kg = nx.DiGraph()

# 添加药物-功效边
herb_efficacy_count = defaultdict(lambda: defaultdict(int))
for _, row in df.iterrows():
    herb = row['名称']
    efficacy = get_herb_efficacy(herb)
    herb_efficacy_count[herb][efficacy] += 1

# 取高频药物（出现>50次）
drug_freq = df['名称'].value_counts()
top_drugs = drug_freq[drug_freq >= 30].index.tolist()

for herb in top_drugs:
    efficacy = get_herb_efficacy(herb)
    if efficacy != '其他':
        G_kg.add_node(herb, type='drug', freq=int(drug_freq[herb]))
        G_kg.add_node(efficacy, type='efficacy')
        G_kg.add_edge(herb, efficacy, weight=int(herb_efficacy_count[herb][efficacy]))

# 添加功效-证型边
for herb in top_drugs:
    efficacy = get_herb_efficacy(herb)
    if efficacy == '其他':
        continue
    for syndrome, count in herb_syndrome[herb].items():
        if count >= 5:
            G_kg.add_node(syndrome, type='syndrome')
            if G_kg.has_node(efficacy):
                G_kg.add_edge(efficacy, syndrome, weight=count)

# 添加证型-疾病边
top_syndromes = df['主要证型'].value_counts().head(15).index.tolist()
for syndrome in top_syndromes:
    for disease, count in syndrome_disease[syndrome].items():
        if count >= 5:
            G_kg.add_node(disease, type='disease')
            if G_kg.has_node(syndrome):
                G_kg.add_edge(syndrome, disease, weight=count)

print(f"  知识图谱: {G_kg.number_of_nodes()}个节点, {G_kg.number_of_edges()}条边")

# 可视化知识图谱（分层布局）
fig, ax = plt.subplots(figsize=(20, 14))

node_types = nx.get_node_attributes(G_kg, 'type')
type_colors = {'drug': '#FF6B6B', 'efficacy': '#4ECDC4', 'syndrome': '#45B7D1', 'disease': '#96CEB4'}
type_labels = {'drug': '药物', 'efficacy': '功效', 'syndrome': '证型', 'disease': '疾病'}

# 分层布局
pos = {}
layer_x = {'drug': 0, 'efficacy': 3, 'syndrome': 6, 'disease': 9}
layer_counts = defaultdict(int)

for node in G_kg.nodes():
    ntype = node_types.get(node, 'drug')
    layer_counts[ntype] += 1

layer_y_offset = {}
for ntype in ['drug', 'efficacy', 'syndrome', 'disease']:
    layer_y_offset[ntype] = 0

for node in G_kg.nodes():
    ntype = node_types.get(node, 'drug')
    total_in_layer = layer_counts[ntype]
    y = (layer_y_offset[ntype] / max(total_in_layer - 1, 1) - 0.5) * 20 if total_in_layer > 1 else 0
    pos[node] = (layer_x[ntype], y)
    layer_y_offset[ntype] += 1

# 绘制边
for u, v, data in G_kg.edges(data=True):
    weight = data.get('weight', 1)
    alpha = min(0.3 + weight / 100, 0.8)
    lw = min(0.5 + weight / 50, 3)
    ax.annotate('', xy=pos[v], xytext=pos[u],
               arrowprops=dict(arrowstyle='->', color='gray', alpha=alpha, lw=lw))

# 绘制节点
for ntype, color in type_colors.items():
    nodes = [n for n in G_kg.nodes() if node_types.get(n) == ntype]
    if not nodes:
        continue
    sizes = []
    for n in nodes:
        if ntype == 'drug':
            sizes.append(G_kg.nodes[n].get('freq', 10) * 2)
        else:
            sizes.append(300)
    nx.draw_networkx_nodes(G_kg, pos, nodelist=nodes, node_color=color,
                          node_size=sizes, alpha=0.8, ax=ax)
    for n in nodes:
        ax.annotate(n, pos[n], fontsize=6, ha='center', va='center', fontweight='bold')

# 图例
for ntype, color in type_colors.items():
    ax.scatter([], [], c=color, s=100, label=type_labels[ntype])
ax.legend(fontsize=12, loc='upper right', title='知识图谱层次', title_fontsize=13)
ax.set_title('赵汉青教授中药知识图谱（药物→功效→证型→疾病）', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_knowledge_graph.png')
plt.close()
print("  [图] 02_knowledge_graph.png 已保存")

# 核心用药路径：找出高频的 drug->efficacy->syndrome->disease 路径
core_paths = []
for herb in top_drugs[:20]:
    efficacy = get_herb_efficacy(herb)
    if efficacy == '其他':
        continue
    for syndrome, scount in sorted(herb_syndrome[herb].items(), key=lambda x: -x[1])[:3]:
        for disease, dcount in sorted(syndrome_disease[syndrome].items(), key=lambda x: -x[1])[:2]:
            core_paths.append({
                'path': f'{herb} → {efficacy} → {syndrome} → {disease}',
                'herb': herb, 'efficacy': efficacy,
                'syndrome': syndrome, 'disease': disease,
                'herb_syndrome_count': scount,
                'syndrome_disease_count': dcount
            })

core_paths.sort(key=lambda x: -x['herb_syndrome_count'])
print(f"  核心用药路径 (前10):")
for p in core_paths[:10]:
    print(f"    {p['path']} (药-证:{p['herb_syndrome_count']}, 证-病:{p['syndrome_disease_count']})")

results['knowledge_graph'] = {
    'nodes': G_kg.number_of_nodes(),
    'edges': G_kg.number_of_edges(),
    'core_paths': core_paths[:20]
}


# ============================================================
# 3. 季节用药规律分析
# ============================================================
print("\n" + "=" * 60)
print("3. 季节用药规律分析")
print("=" * 60)

seasons = ['春', '夏', '秋', '冬']
season_drug_matrix = pd.DataFrame(0, index=seasons,
                                   columns=df['名称'].value_counts().head(50).index)

for _, row in df.iterrows():
    herb = row['名称']
    season = row['季节']
    if herb in season_drug_matrix.columns and season in season_drug_matrix.index:
        season_drug_matrix.loc[season, herb] += 1

# 卡方检验：每个药物的季节分布是否显著偏离均匀
chi2_results = []
season_totals = df.groupby('季节').size()
total_records = len(df)

for herb in season_drug_matrix.columns:
    observed = season_drug_matrix[herb].values
    if observed.sum() < 20:
        continue
    expected = season_totals.values * (observed.sum() / total_records)
    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    chi2_results.append({
        'herb': herb,
        'chi2': round(chi2, 2),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'spring': int(observed[0]),
        'summer': int(observed[1]),
        'autumn': int(observed[2]),
        'winter': int(observed[3]),
        'total': int(observed.sum())
    })

chi2_df = pd.DataFrame(chi2_results)
sig_drugs = chi2_df[chi2_df['significant']].sort_values('chi2', ascending=False)
print(f"  季节差异显著药物(p<0.05): {len(sig_drugs)}种")
print(f"  Top 10:")
for _, r in sig_drugs.head(10).iterrows():
    print(f"    {r['herb']}: χ²={r['chi2']}, p={r['p_value']} "
          f"(春{r['spring']}/夏{r['summer']}/秋{r['autumn']}/冬{r['winter']})")

# 季节特异性药对
season_pairs = {}
for season in seasons:
    season_df = df[df['季节'] == season]
    prescriptions = season_df.groupby('处方号')['名称'].apply(set)
    pair_count = Counter()
    for drugs in prescriptions:
        for pair in itertools.combinations(sorted(drugs), 2):
            pair_count[pair] += 1
    # 找出该季节特有的高频药对
    top_pairs = pair_count.most_common(20)
    season_pairs[season] = [(f'{p[0][0]}+{p[0][1]}', p[1]) for p in top_pairs]

print(f"\n  各季节高频药对:")
for season in seasons:
    print(f"    {season}季: {', '.join([f'{p[0]}({p[1]})' for p in season_pairs[season][:5]])}")

# 绘制季节热力图
fig, ax = plt.subplots(figsize=(18, 8))
top_sig = sig_drugs.head(30)['herb'].tolist() if len(sig_drugs) >= 10 else chi2_df.head(30)['herb'].tolist()
if top_sig:
    plot_data = season_drug_matrix[top_sig].T
    # 标准化为行百分比
    plot_data_pct = plot_data.div(plot_data.sum(axis=1), axis=0) * 100

    im = ax.imshow(plot_data_pct.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(seasons, fontsize=12)
    ax.set_yticks(range(len(top_sig)))
    ax.set_yticklabels(top_sig, fontsize=8)
    ax.set_title('季节-药物使用频率热力图（行标准化%）', fontsize=14, fontweight='bold')
    ax.set_xlabel('季节', fontsize=12)
    plt.colorbar(im, label='使用频率(%)')

    # 标注数值
    for i in range(len(top_sig)):
        for j in range(4):
            val = plot_data_pct.iloc[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6,
                   color='white' if val > 50 else 'black')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_season_heatmap.png')
plt.close()
print("  [图] 03_season_heatmap.png 已保存")

results['season_analysis'] = {
    'significant_drugs': len(sig_drugs),
    'top_seasonal_drugs': sig_drugs.head(15).to_dict('records') if not sig_drugs.empty else [],
    'season_pairs': season_pairs
}


# ============================================================
# 4. 剂量规律分析
# ============================================================
print("\n" + "=" * 60)
print("4. 剂量规律分析")
print("=" * 60)

# 同一药物在不同证型下的剂量差异
dose_analysis = []
top20_drugs = df['名称'].value_counts().head(30).index.tolist()

for herb in top20_drugs:
    herb_data = df[df['名称'] == herb]
    syndromes = herb_data['主要证型'].value_counts().head(5).index
    doses_by_syndrome = {}
    for syn in syndromes:
        doses = herb_data[herb_data['主要证型'] == syn]['每次用量']
        if len(doses) >= 5:
            doses_by_syndrome[syn] = {
                'mean': round(float(doses.mean()), 1),
                'std': round(float(doses.std()), 1),
                'min': float(doses.min()),
                'max': float(doses.max()),
                'count': int(len(doses))
            }

    overall = herb_data['每次用量']
    dose_analysis.append({
        'herb': herb,
        'overall_mean': round(float(overall.mean()), 1),
        'overall_std': round(float(overall.std()), 1),
        'overall_median': float(overall.median()),
        'by_syndrome': doses_by_syndrome
    })

print(f"  分析了{len(dose_analysis)}种药物的剂量规律")
for d in dose_analysis[:5]:
    print(f"    {d['herb']}: 均值{d['overall_mean']}g, 中位{d['overall_median']}g, "
          f"范围{d['overall_std']}g")

# 剂量聚类
dose_features = []
dose_herb_names = []
for herb in top20_drugs:
    herb_doses = df[df['名称'] == herb]['每次用量']
    dose_features.append([
        herb_doses.mean(),
        herb_doses.std(),
        herb_doses.median(),
        herb_doses.min(),
        herb_doses.max(),
        (herb_doses > herb_doses.median()).sum() / len(herb_doses)  # 大剂量比例
    ])
    dose_herb_names.append(herb)

dose_features = np.array(dose_features)
# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dose_features_scaled = scaler.fit_transform(dose_features)

# 层次聚类
Z = linkage(dose_features_scaled, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')

cluster_labels = {1: '小剂量组', 2: '常规剂量组', 3: '大剂量组'}
dose_clusters = defaultdict(list)
for herb, cluster_id in zip(dose_herb_names, clusters):
    mean_dose = df[df['名称'] == herb]['每次用量'].mean()
    dose_clusters[cluster_id].append((herb, round(mean_dose, 1)))

# 重新按平均剂量标记
cluster_means = {}
for cid in dose_clusters:
    avg = np.mean([d[1] for d in dose_clusters[cid]])
    cluster_means[cid] = avg

sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
label_map = {sorted_clusters[0][0]: '小剂量组',
             sorted_clusters[1][0]: '常规剂量组',
             sorted_clusters[2][0]: '大剂量组'}

print(f"\n  剂量聚类结果:")
for cid in dose_clusters:
    label = label_map.get(cid, f'组{cid}')
    herbs = dose_clusters[cid]
    print(f"    {label} (均值{cluster_means[cid]:.1f}g): "
          f"{', '.join([f'{h}({d}g)' for h, d in herbs[:8]])}")

# 绘制剂量聚类树状图
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 左图：树状图
dendrogram(Z, labels=dose_herb_names, ax=axes[0], leaf_font_size=7,
           color_threshold=Z[-3, 2] if len(Z) >= 3 else 0)
axes[0].set_title('药物剂量层次聚类树状图', fontsize=14, fontweight='bold')
axes[0].set_ylabel('距离', fontsize=12)

# 右图：箱线图（选代表药物）
representative_herbs = []
for cid in dose_clusters:
    label = label_map.get(cid, f'组{cid}')
    for herb, _ in dose_clusters[cid][:4]:
        representative_herbs.append((herb, label))

rep_data = [(df[df['名称'] == h]['每次用量'].values, h, l) for h, l in representative_herbs[:16]]
positions = range(len(rep_data))
bp = axes[1].boxplot([d[0] for d in rep_data], positions=positions, patch_artist=True,
                      widths=0.6)
colors_box = []
for _, _, label in rep_data:
    if '小' in label:
        colors_box.append('#81C784')
    elif '大' in label:
        colors_box.append('#E57373')
    else:
        colors_box.append('#64B5F6')
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
axes[1].set_xticks(positions)
axes[1].set_xticklabels([d[1] for d in rep_data], rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('每次用量(g)', fontsize=12)
axes[1].set_title('药物剂量分布箱线图（按聚类着色）', fontsize=14, fontweight='bold')

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#81C784', label='小剂量组'),
                   Patch(facecolor='#64B5F6', label='常规剂量组'),
                   Patch(facecolor='#E57373', label='大剂量组')]
axes[1].legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_dose_clustering.png')
plt.close()
print("  [图] 04_dose_clustering.png 已保存")

results['dose_analysis'] = {
    'top_drug_doses': dose_analysis[:20],
    'dose_clusters': {
        label_map.get(cid, f'组{cid}'): {
            'avg_dose': round(cluster_means[cid], 1),
            'herbs': [h for h, _ in dose_clusters[cid]]
        } for cid in dose_clusters
    }
}


# ============================================================
# 5. 治法反推分析
# ============================================================
print("\n" + "=" * 60)
print("5. 治法反推分析")
print("=" * 60)

# 反推每张处方的治法
prescription_methods = {}
method_freq = Counter()
method_disease = defaultdict(lambda: Counter())

for presc_id in df['处方号'].unique():
    presc_drugs = df[df['处方号'] == presc_id]['名称'].unique()
    presc_syndrome = df[df['处方号'] == presc_id]['主要证型'].iloc[0]
    presc_disease = df[df['处方号'] == presc_id]['主要疾病'].iloc[0]

    methods = set()
    for herb in presc_drugs:
        efficacy = get_herb_efficacy(herb)
        method = get_treatment_method(efficacy)
        if method != '其他治法':
            methods.add(method)

    prescription_methods[presc_id] = sorted(list(methods))

    for method in methods:
        method_freq[method] += 1
        if presc_disease:
            method_disease[method][presc_disease] += 1

print(f"  识别出{len(method_freq)}种治法")
print(f"  治法频次Top 15:")
for method, count in method_freq.most_common(15):
    print(f"    {method}: {count}次 ({count/len(prescription_methods)*100:.1f}%)")

# 治法组合分析（每张处方的治法组合）
method_combo_freq = Counter()
for presc_id, methods in prescription_methods.items():
    if len(methods) >= 2:
        for combo in itertools.combinations(sorted(methods), 2):
            method_combo_freq[combo] += 1

print(f"\n  高频治法组合Top 10:")
for combo, count in method_combo_freq.most_common(10):
    print(f"    {combo[0]} + {combo[1]}: {count}次")

# 治法-疾病关联
print(f"\n  各治法对应Top疾病:")
method_disease_top = {}
for method in [m for m, _ in method_freq.most_common(10)]:
    top_diseases = method_disease[method].most_common(5)
    method_disease_top[method] = [(d, int(c)) for d, c in top_diseases]
    print(f"    {method}: {', '.join([f'{d}({c})' for d, c in top_diseases[:3]])}")

# 绘制治法图谱
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 左图：治法频次柱状图
top_methods = method_freq.most_common(20)
method_names = [m[0] for m in top_methods]
method_counts = [m[1] for m in top_methods]
colors_m = plt.cm.viridis(np.linspace(0.2, 0.8, len(method_names)))
axes[0].barh(range(len(method_names)), method_counts, color=colors_m)
axes[0].set_yticks(range(len(method_names)))
axes[0].set_yticklabels(method_names, fontsize=10)
axes[0].set_xlabel('出现处方数', fontsize=12)
axes[0].set_title('治法频次分布', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# 右图：治法网络图
G_method = nx.Graph()
for combo, count in method_combo_freq.most_common(50):
    if count >= 5:
        G_method.add_edge(combo[0], combo[1], weight=count)

if G_method.number_of_nodes() > 0:
    pos_m = nx.spring_layout(G_method, k=2, seed=42)
    node_sizes = [method_freq.get(n, 10) * 8 for n in G_method.nodes()]
    edge_weights = [G_method[u][v]['weight'] / 5 for u, v in G_method.edges()]

    nx.draw_networkx_nodes(G_method, pos_m, node_size=node_sizes,
                          node_color='lightcoral', alpha=0.7, ax=axes[1])
    nx.draw_networkx_edges(G_method, pos_m, width=edge_weights,
                          alpha=0.5, edge_color='gray', ax=axes[1])
    nx.draw_networkx_labels(G_method, pos_m, font_size=8,
                           font_family=FONT_NAME, ax=axes[1])

axes[1].set_title('治法关联网络图', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_treatment_methods.png')
plt.close()
print("  [图] 05_treatment_methods.png 已保存")

results['treatment_methods'] = {
    'method_frequency': dict(method_freq.most_common(20)),
    'method_combinations': [
        {'methods': f'{c[0]}+{c[1]}', 'count': cnt}
        for c, cnt in method_combo_freq.most_common(20)
    ],
    'method_disease': method_disease_top
}


# ============================================================
# 6. 核心处方发现算法
# ============================================================
print("\n" + "=" * 60)
print("6. 核心处方发现算法")
print("=" * 60)

# 构建药物共现网络
prescriptions_drugsets = df.groupby('处方号')['名称'].apply(set)
co_occurrence = Counter()
for drugs in prescriptions_drugsets:
    for pair in itertools.combinations(sorted(drugs), 2):
        co_occurrence[pair] += 1

print(f"  药物共现对数: {len(co_occurrence)}")

# 频繁子图挖掘：找高频药物组合（从2味到6味）
frequent_itemsets = {}
for size in range(2, 7):
    itemset_count = Counter()
    for drugs in prescriptions_drugsets:
        if len(drugs) >= size:
            for combo in itertools.combinations(sorted(drugs), size):
                itemset_count[combo] += 1
    # 支持度阈值：出现在至少5%的处方中
    min_support = max(5, len(prescriptions_drugsets) * 0.03)
    frequent = {k: v for k, v in itemset_count.items() if v >= min_support}
    frequent_itemsets[size] = sorted(frequent.items(), key=lambda x: -x[1])
    print(f"  {size}味药组合 (≥{min_support}次): {len(frequent)}个")

# 核心方剂模板：3味以上高频组合
core_formulas = []
for size in range(6, 2, -1):
    for combo, count in frequent_itemsets.get(size, [])[:10]:
        core_formulas.append({
            'drugs': list(combo),
            'size': size,
            'count': count,
            'support': round(count / len(prescriptions_drugsets) * 100, 1)
        })

print(f"\n  核心方剂模板Top 10:")
for f in core_formulas[:10]:
    print(f"    {f['size']}味 ({f['support']}%): {', '.join(f['drugs'])} ({f['count']}次)")

# 与经典方剂对比
formula_comparison = []
for formula_name, classic_drugs in CLASSIC_FORMULAS.items():
    max_overlap = 0
    best_match = None
    for combo, count in co_occurrence.most_common(1000):
        combo_set = set(combo)
        overlap = len(classic_drugs & combo_set)
        overlap_rate = overlap / len(classic_drugs) if classic_drugs else 0
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = (combo_set, count, overlap_rate)

    if best_match:
        formula_comparison.append({
            'classic_formula': formula_name,
            'classic_drugs': sorted(list(classic_drugs)),
            'match_drugs': sorted(list(best_match[0] & classic_drugs)),
            'overlap_count': max_overlap,
            'overlap_rate': round(max_overlap / len(classic_drugs) * 100, 1) if classic_drugs else 0,
            'co_occurrence': best_match[1]
        })

formula_comparison.sort(key=lambda x: -x['overlap_rate'])
print(f"\n  与经典方剂对比:")
for fc in formula_comparison[:10]:
    print(f"    {fc['classic_formula']}: 匹配{fc['overlap_count']}/{len(fc['classic_drugs'])}味 "
          f"({fc['overlap_rate']}%) = {', '.join(fc['match_drugs'])}")

# 绘制核心处方发现图
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 左图：药物共现网络（高频）
G_co = nx.Graph()
for pair, count in co_occurrence.most_common(100):
    if count >= 10:
        G_co.add_edge(pair[0], pair[1], weight=count)

if G_co.number_of_nodes() > 0:
    pos_co = nx.spring_layout(G_co, k=1.5, seed=42)
    node_freq_co = {n: df[df['名称'] == n].shape[0] for n in G_co.nodes()}
    node_sizes_co = [node_freq_co.get(n, 10) * 3 for n in G_co.nodes()]
    edge_widths_co = [G_co[u][v]['weight'] / 10 for u, v in G_co.edges()]

    nx.draw_networkx_nodes(G_co, pos_co, node_size=node_sizes_co,
                          node_color='salmon', alpha=0.7, ax=axes[0])
    nx.draw_networkx_edges(G_co, pos_co, width=edge_widths_co,
                          alpha=0.3, ax=axes[0])
    # 只标注度数最高的节点
    degree_dict = dict(G_co.degree())
    top_labels = sorted(degree_dict, key=degree_dict.get, reverse=True)[:30]
    labels_co = {n: n for n in top_labels if n in G_co.nodes()}
    nx.draw_networkx_labels(G_co, pos_co, labels=labels_co, font_size=7,
                           font_family=FONT_NAME, ax=axes[0])

axes[0].set_title('药物共现网络（高频100对）', fontsize=14, fontweight='bold')
axes[0].axis('off')

# 右图：经典方剂匹配雷达图/对比图
if formula_comparison:
    fc_top = formula_comparison[:8]
    names = [fc['classic_formula'] for fc in fc_top]
    rates = [fc['overlap_rate'] for fc in fc_top]
    colors_fc = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = axes[1].barh(range(len(names)), rates, color=colors_fc, edgecolor='gray')
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=11)
    axes[1].set_xlabel('药物匹配率(%)', fontsize=12)
    axes[1].set_title('处方模板与经典方剂匹配度', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    for bar, rate, fc in zip(bars, rates, fc_top):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{rate}% ({fc["overlap_count"]}/{len(fc["classic_drugs"])})',
                    va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_core_prescriptions.png')
plt.close()
print("  [图] 06_core_prescriptions.png 已保存")

results['core_prescriptions'] = {
    'frequent_itemsets': {
        str(k): [{'drugs': list(combo), 'count': cnt}
                 for combo, cnt in v[:10]]
        for k, v in frequent_itemsets.items()
    },
    'core_formulas': core_formulas[:15],
    'classic_formula_comparison': formula_comparison
}


# ============================================================
# 保存结果
# ============================================================
print("\n" + "=" * 60)
print("保存结果")
print("=" * 60)

# 确保所有值可序列化
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj

results = make_serializable(results)

with open(RES_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"  结果已保存到 {RES_FILE}")

print("\n" + "=" * 60)
print("创新分析完成！")
print("=" * 60)
print(f"""
生成图表:
  1. figures/01_prescription_evolution.png - 处方演化分析
  2. figures/01b_evolution_sankey.png - 处方演化桑基图
  3. figures/02_knowledge_graph.png - 中药知识图谱
  4. figures/03_season_heatmap.png - 季节用药热力图
  5. figures/04_dose_clustering.png - 剂量聚类分析
  6. figures/05_treatment_methods.png - 治法分析
  7. figures/06_core_prescriptions.png - 核心处方发现

结果数据: results/innovation_results.json
""")
