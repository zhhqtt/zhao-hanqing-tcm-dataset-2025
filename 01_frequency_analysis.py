#!/usr/bin/env python3
"""
赵汉青国医大师门诊方药数据挖掘 - 频次分析与描述性统计
"""
import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter

warnings.filterwarnings('ignore')

# ── 字体设置 ──
for fname in ['SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']:
    try:
        fp = font_manager.findfont(fname, fallback_to_default=False)
        if fp:
            plt.rcParams['font.sans-serif'] = [fname, 'DejaVu Sans']
            break
    except:
        continue
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'figures')
RES = os.path.join(BASE, 'results')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

df = pd.read_pickle(os.path.join(BASE, 'data', 'cleaned_data.pkl'))
results = {}

# ── 辅助 ──
def save_json(obj, path):
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NpEncoder)

def barh_top(series, top, title, xlabel, filename, figsize=None):
    figsize = figsize or (10, max(6, top*0.35))
    fig, ax = plt.subplots(figsize=figsize)
    s = series.value_counts().head(top)
    bars = ax.barh(range(len(s)), s.values, color='steelblue')
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(s.values):
        ax.text(v + s.values.max()*0.01, i, f'{v} ({v/series.notna().sum()*100:.1f}%)', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, filename), dpi=150, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════
# 1. 患者人口学统计
# ═══════════════════════════════════════════
print("1. 患者人口学统计...")

# 患者级数据（去重）
patients = df.drop_duplicates(subset=['病历号\u3000']).copy() if '病历号\u3000' in df.columns else df.drop_duplicates(subset=['姓名','性别','年龄'])

gender = df.drop_duplicates(subset='处方号')['性别'].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
gender.plot(kind='bar', color=['#FF9999','#66B2FF'], ax=ax)
ax.set_title('处方患者性别分布')
ax.set_ylabel('处方数')
for i, v in enumerate(gender.values):
    ax.text(i, v+5, f'{v} ({v/gender.sum()*100:.1f}%)', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '01_gender_distribution.png'), dpi=150)
plt.close()

# 年龄分布
age_series = df.drop_duplicates(subset='处方号')['年龄'].dropna()
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(age_series, bins=20, color='steelblue', edgecolor='white')
ax.set_title('患者年龄分布')
ax.set_xlabel('年龄')
ax.set_ylabel('处方数')
ax.axvline(age_series.mean(), color='red', linestyle='--', label=f'均值={age_series.mean():.1f}岁')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, '02_age_distribution.png'), dpi=150)
plt.close()

age_stats = {'mean': float(age_series.mean()), 'median': float(age_series.median()),
             'std': float(age_series.std()), 'min': float(age_series.min()), 'max': float(age_series.max())}

# 初复诊
visit_type = df.drop_duplicates(subset='处方号')['初复诊'].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
visit_type.plot(kind='bar', color=['#99CC99','#FFCC99'], ax=ax)
ax.set_title('初复诊分布')
ax.set_ylabel('处方数')
for i, v in enumerate(visit_type.values):
    ax.text(i, v+5, f'{v} ({v/visit_type.sum()*100:.1f}%)', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '03_visit_type.png'), dpi=150)
plt.close()

# 就诊频次
patient_visits = df.drop_duplicates(subset='处方号').groupby('姓名').size()
fig, ax = plt.subplots(figsize=(8, 5))
bins = list(range(1, patient_visits.max()+2))
ax.hist(patient_visits, bins=min(30, len(bins)), color='steelblue', edgecolor='white')
ax.set_title('患者就诊频次分布')
ax.set_xlabel('就诊次数')
ax.set_ylabel('患者数')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '04_visit_frequency.png'), dpi=150)
plt.close()

results['demographics'] = {
    'total_records': len(df),
    'total_prescriptions': int(df['处方号'].nunique()),
    'total_patients': int(df['姓名'].nunique()),
    'gender_by_prescription': gender.to_dict(),
    'age_stats': age_stats,
    'visit_type': visit_type.to_dict(),
    'visit_frequency_stats': {
        'mean': float(patient_visits.mean()),
        'median': float(patient_visits.median()),
        'max': int(patient_visits.max())
    }
}

# ═══════════════════════════════════════════
# 2. 药物频次分析
# ═══════════════════════════════════════════
print("2. 药物频次分析...")

herb_freq = df['名称'].value_counts()
total_rx = df['处方号'].nunique()

# Top50
fig, ax = plt.subplots(figsize=(12, 18))
top50 = herb_freq.head(50)
bars = ax.barh(range(len(top50)), top50.values, color='steelblue')
ax.set_yticks(range(len(top50)))
ax.set_yticklabels(top50.index)
ax.invert_yaxis()
ax.set_xlabel('出现频次')
ax.set_title('药物频次Top50')
for i, v in enumerate(top50.values):
    ax.text(v + top50.values.max()*0.01, i, f'{v}({v/total_rx*100:.1f}%)', va='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG, '05_herb_frequency_top50.png'), dpi=150)
plt.close()

# 药物分类映射
HERB_CATEGORY = {
    '补气药': ['黄芪','党参','白术','甘草','炙甘草','太子参','山药','白扁豆','大枣','蜂蜜','绞股蓝','刺五加','红景天','红参','人参','西洋参','白芍','饴糖'],
    '补血药': ['当归','熟地黄','白芍','阿胶','何首乌','龙眼肉','鸡血藤','枸杞子','桑椹'],
    '补阴药': ['麦冬','北沙参','南沙参','石斛','玉竹','百合','黄精','女贞子','墨旱莲','龟甲','鳖甲','天冬','明党参'],
    '补阳药': ['杜仲','续断','菟丝子','肉苁蓉','淫羊藿','巴戟天','补骨脂','益智仁','仙茅','骨碎补','狗脊','鹿角胶','鹿角霜','蛤蚧','冬虫夏草','胡桃仁','葫芦巴','沙苑子','韭菜子','蛇床子'],
    '活血化瘀药': ['丹参','川芎','红花','桃仁','赤芍','牛膝','鸡血藤','延胡索','郁金','姜黄','乳香','没药','五灵脂','三棱','莪术','水蛭','穿山甲','王不留行','泽兰','益母草','苏木','降香','自然铜','骨碎补','血竭','土鳖虫','虻虫','斑蝥'],
    '理气药': ['陈皮','枳实','枳壳','木香','香附','乌药','沉香','檀香','川楝子','青皮','佛手','香橼','玫瑰花','绿萼梅','大腹皮','薤白','甘松','荔枝核','九香虫','柿蒂'],
    '清热药': ['黄芩','黄连','黄柏','栀子','石膏','知母','天花粉','芦根','夏枯草','决明子','龙胆草','苦参','金银花','连翘','蒲公英','紫花地丁','大青叶','板蓝根','鱼腥草','败酱草','白花蛇舌草','半枝莲','射干','山豆根','马勃','青黛','白头翁','马齿苋','鸦胆子','地骨皮','银柴胡','胡黄连','青蒿','白薇'],
    '祛湿药/利水渗湿药': ['茯苓','猪苓','泽泻','薏苡仁','车前子','滑石','木通','通草','瞿麦','萹蓄','地肤子','海金沙','石韦','萆薢','茵陈','金钱草','虎杖','地耳草','垂盆草','土茯苓','半边莲','冬瓜皮','玉米须'],
    '温里药': ['附子','干姜','肉桂','吴茱萸','小茴香','丁香','高良姜','胡椒','花椒','荜茇','荜澄茄'],
    '化痰止咳平喘药': ['半夏','天南星','白芥子','桔梗','旋覆花','白前','前胡','川贝母','浙贝母','瓜蒌','杏仁','紫苏子','百部','紫菀','款冬花','枇杷叶','桑白皮','葶苈子','海藻','昆布','蛤壳','浮海石','瓦楞子','礞石','胖大海','罗汉果'],
    '解表药': ['麻黄','桂枝','紫苏叶','荆芥','防风','羌活','白芷','细辛','藁本','苍耳子','辛夷','生姜','葱白','薄荷','牛蒡子','蝉蜕','桑叶','菊花','蔓荆子','柴胡','葛根','升麻','浮萍','淡豆豉'],
    '平肝熄风药': ['天麻','钩藤','石决明','珍珠母','牡蛎','代赭石','刺蒺藜','罗布麻','地龙','全蝎','蜈蚣','僵蚕','珍珠','羚羊角'],
    '安神药': ['酸枣仁','柏子仁','远志','合欢皮','合欢花','龙骨','琥珀','朱砂','磁石','夜交藤'],
    '收涩药': ['五味子','乌梅','五倍子','罂粟壳','诃子','肉豆蔻','赤石脂','禹余粮','山茱萸','覆盆子','桑螵蛸','海螵蛸','莲子','芡实','金樱子','椿皮','石榴皮'],
    '消食药': ['山楂','神曲','麦芽','谷芽','莱菔子','鸡内金'],
    '祛风湿药': ['独活','威灵仙','川乌','草乌','蕲蛇','乌梢蛇','木瓜','络石藤','徐长卿','桑枝','桑寄生','五加皮','千年健','秦艽','防己','豨莶草','臭梧桐','海风藤','青风藤','雷公藤','穿山龙'],
    '止血药': ['三七','蒲黄','白及','仙鹤草','棕榈炭','血余炭','藕节','艾叶','灶心土','侧柏叶','白茅根','槐花','地榆','茜草','花蕊石'],
    '泻下药': ['大黄','芒硝','番泻叶','芦荟','火麻仁','郁李仁','甘遂','大戟','芫花','牵牛子'],
    '化湿药': ['苍术','厚朴','藿香','佩兰','砂仁','白豆蔻','草豆蔻','草果'],
    '开窍药': ['麝香','冰片','苏合香','石菖蒲','蟾酥'],
}

cat_counts = {}
for cat, herbs in HERB_CATEGORY.items():
    cnt = herb_freq[herb_freq.index.isin(herbs)].sum()
    if cnt > 0:
        cat_counts[cat] = int(cnt)

cat_series = pd.Series(cat_counts).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(cat_series)), cat_series.values, color='steelblue')
ax.set_yticks(range(len(cat_series)))
ax.set_yticklabels(cat_series.index)
ax.set_xlabel('出现频次')
ax.set_title('药物功效分类频次')
for i, v in enumerate(cat_series.values):
    ax.text(v + cat_series.values.max()*0.01, i, str(v), va='center')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '06_herb_category.png'), dpi=150)
plt.close()

# 每张处方平均药味数
rx_herb_count = df.groupby('处方号')['名称'].nunique()
avg_herbs = float(rx_herb_count.mean())

# 单味药剂量分布Top20
dose = df[df['单位']=='g'].groupby('名称')['每次用量'].agg(['mean','median','std','count']).sort_values('count', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(dose))
ax.bar(x, dose['mean'], yerr=dose['std'].fillna(0), color='steelblue', capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(dose.index, rotation=45, ha='right')
ax.set_ylabel('平均剂量(g)')
ax.set_title('Top20高频药物平均剂量(g)±标准差')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '07_herb_dose_top20.png'), dpi=150)
plt.close()

results['herb'] = {
    'total_herbs': int(df['名称'].nunique()),
    'top50': {k: {'count': int(v), 'percentage': round(v/total_rx*100, 2)} for k, v in top50.items()},
    'category': cat_counts,
    'avg_herbs_per_prescription': round(avg_herbs, 2),
    'herbs_per_prescription_stats': {
        'mean': float(rx_herb_count.mean()),
        'median': float(rx_herb_count.median()),
        'min': int(rx_herb_count.min()),
        'max': int(rx_herb_count.max()),
        'std': float(rx_herb_count.std()),
    },
    'dose_top20': {name: {'mean': round(row['mean'],2), 'median': round(row['median'],2), 'count': int(row['count'])} for name, row in dose.iterrows()},
}

# ═══════════════════════════════════════════
# 3. 证型频次分析
# ═══════════════════════════════════════════
print("3. 证型频次分析...")

syndrome_freq = df.drop_duplicates(subset='处方号')['主要证型'].value_counts()
barh_top(df.drop_duplicates(subset='处方号')['主要证型'], 30,
         '证型频次Top30', '处方数', '08_syndrome_top30.png')

# 证型归类
def classify_syndrome(s):
    if pd.isna(s): return '未知'
    s = str(s)
    has_deficiency = any(x in s for x in ['虚','不足','亏','衰','弱'])
    has_excess = any(x in s for x in ['瘀','痰','湿','热','寒','郁','滞','火','毒','结','积','饮','水','风','闭'])
    if has_deficiency and has_excess: return '虚实夹杂'
    if has_deficiency: return '虚证'
    if has_excess: return '实证'
    return '其他'

syn_clf = df.drop_duplicates(subset='处方号')['主要证型'].apply(classify_syndrome).value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
syn_clf.plot(kind='bar', color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'], ax=ax)
ax.set_title('证型虚实分类')
ax.set_ylabel('处方数')
for i, v in enumerate(syn_clf.values):
    ax.text(i, v+3, f'{v} ({v/syn_clf.sum()*100:.1f}%)', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '09_syndrome_classification.png'), dpi=150)
plt.close()

results['syndrome'] = {
    'total_types': int(syndrome_freq.shape[0]),
    'top30': syndrome_freq.head(30).to_dict(),
    'deficiency_excess': syn_clf.to_dict(),
}

# ═══════════════════════════════════════════
# 4. 疾病频次分析
# ═══════════════════════════════════════════
print("4. 疾病频次分析...")

disease_freq = df.drop_duplicates(subset='处方号')['主要疾病'].value_counts()
barh_top(df.drop_duplicates(subset='处方号')['主要疾病'], 30,
         '疾病频次Top30', '处方数', '10_disease_top30.png')

# 疾病分类
def classify_disease(d):
    if pd.isna(d): return '未知'
    d = str(d)
    gynecology = ['月经','子宫','卵巢','盆腔','乳腺','更年期','闭经','痛经','带下','胎','产','孕','不孕','流产','崩漏','阴痒','阴挺','阴疮']
    orthopedics = ['骨','关节','颈','腰椎','肩','膝','腰痛','跌','骨折','软组织','扭','损伤','痛风','风湿','痹']
    internal = ['胃','肠','肝','胆','脾','肺','心','肾','咳','喘','哮','消渴','糖尿','高血','冠心','心悸','胸痹','中风','头痛','眩晕','失眠','不寐','胃痛','痞满','呕吐','泄泻','便秘','腹痛','黄疸','鼓胀','水肿','淋证','癃闭','阳痿','遗精','早泄','郁证','癫','痫','痴呆','汗证','虚劳','血证','癌','瘤','结直肠癌','肺癌','胃癌','肝癌','甲状腺','甲亢','甲减','结节','息肉','囊肿','贫血','紫癜','发热','感冒','咳嗽']
    ent = ['耳','鼻','喉','咽','扁桃','鼻窦','鼻渊','耳鸣','耳聋','喉痹','乳蛾','喉喑']
    dermatology = ['湿疹','荨麻','银屑','痤疮','疮','癣','疹','疣','疱','风疹','白癜风','黄褐斑','斑','脱发','脂溢性']
    
    if any(x in d for x in gynecology): return '妇科'
    if any(x in d for x in orthopedics): return '骨科'
    if any(x in d for x in dermatology): return '皮肤科'
    if any(x in d for x in ent): return '耳鼻喉科'
    if any(x in d for x in internal): return '内科'
    return '其他'

dis_cat = df.drop_duplicates(subset='处方号')['主要疾病'].apply(classify_disease).value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
dis_cat.plot(kind='bar', color='steelblue', ax=ax)
ax.set_title('疾病科室分类')
ax.set_ylabel('处方数')
for i, v in enumerate(dis_cat.values):
    ax.text(i, v+3, f'{v} ({v/dis_cat.sum()*100:.1f}%)', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '11_disease_category.png'), dpi=150)
plt.close()

results['disease'] = {
    'total_types': int(disease_freq.shape[0]),
    'top30': disease_freq.head(30).to_dict(),
    'department': dis_cat.to_dict(),
}

# ═══════════════════════════════════════════
# 5. 季节分布
# ═══════════════════════════════════════════
print("5. 季节分布...")

rx_unique = df.drop_duplicates(subset='处方号')
monthly = rx_unique.groupby('月份').size()
fig, ax = plt.subplots(figsize=(10, 5))
monthly.plot(kind='line', marker='o', color='steelblue', ax=ax)
ax.fill_between(monthly.index, monthly.values, alpha=0.2)
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f'{m}月' for m in range(1, 13)])
ax.set_ylabel('处方数')
ax.set_title('各月处方量分布')
ax.set_xlim(1, 12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, '12_monthly_prescriptions.png'), dpi=150)
plt.close()

seasonal = rx_unique.groupby('季节').size()
fig, ax = plt.subplots(figsize=(6, 4))
season_order = ['春','夏','秋','冬']
seasonal = seasonal.reindex(season_order).fillna(0)
colors = ['#90EE90','#FFD700','#FF8C00','#87CEEB']
ax.bar(range(len(seasonal)), seasonal.values, color=colors)
ax.set_xticks(range(len(seasonal)))
ax.set_xticklabels(seasonal.index)
ax.set_ylabel('处方数')
ax.set_title('四季处方量分布')
for i, v in enumerate(seasonal.values):
    ax.text(i, v+3, f'{int(v)} ({v/seasonal.sum()*100:.1f}%)', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '13_seasonal_prescriptions.png'), dpi=150)
plt.close()

results['seasonal'] = {
    'monthly': monthly.to_dict(),
    'seasonal': seasonal.to_dict(),
}

# ═══════════════════════════════════════════
# 6. 人群-疾病交叉分析
# ═══════════════════════════════════════════
print("6. 人群-疾病交叉分析...")

# 性别×疾病 Top10
gender_disease = rx_unique.groupby(['性别','主要疾病']).size().unstack(fill_value=0)
top_dis = rx_unique['主要疾病'].value_counts().head(10).index
gd_top = gender_disease[top_dis]
fig, ax = plt.subplots(figsize=(12, 5))
gd_top.T.plot(kind='bar', ax=ax, color=['#FF9999','#66B2FF'])
ax.set_title('性别×疾病Top10')
ax.set_ylabel('处方数')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title='性别')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '14_gender_disease.png'), dpi=150)
plt.close()

# 年龄段×疾病热力图
rx_unique2 = rx_unique.copy()
rx_unique2['年龄段'] = pd.cut(rx_unique2['年龄'], bins=[0,18,30,45,60,75,100],
                              labels=['0-18','19-30','31-45','46-60','61-75','76-100'])
top15_dis = rx_unique2['主要疾病'].value_counts().head(15).index
heat_data = rx_unique2[rx_unique2['主要疾病'].isin(top15_dis)].groupby(['年龄段','主要疾病']).size().unstack(fill_value=0)
heat_data = heat_data.reindex(columns=top15_dis)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(heat_data.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(heat_data.columns)))
ax.set_xticklabels(heat_data.columns, rotation=45, ha='right')
ax.set_yticks(range(len(heat_data.index)))
ax.set_yticklabels(heat_data.index)
ax.set_title('年龄段×疾病Top15 热力图')
plt.colorbar(im, ax=ax, label='处方数')
for i in range(len(heat_data.index)):
    for j in range(len(heat_data.columns)):
        v = heat_data.values[i, j]
        if v > 0:
            ax.text(j, i, str(int(v)), ha='center', va='center', fontsize=7,
                   color='white' if v > heat_data.values.max()*0.6 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '15_age_disease_heatmap.png'), dpi=150)
plt.close()

results['cross_analysis'] = {
    'gender_disease_top10': gd_top.to_dict(),
    'age_disease_heatmap': heat_data.to_dict(),
}

# ═══════════════════════════════════════════
# 保存结果
# ═══════════════════════════════════════════
save_json(results, os.path.join(RES, 'frequency_results.json'))
print(f"\n✅ 分析完成！")
print(f"  图表保存至: {FIG}/")
print(f"  结果保存至: {RES}/frequency_results.json")
print(f"  总计生成图表: {len(os.listdir(FIG))} 张")
