import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, f_oneway, kruskal, chi2_contingency

# 1. 读取数据 (处理好表头后的数据使用 utf-8-sig 读取即可)
df = pd.read_csv('combine-patien-basic-information.cleaned02.csv', encoding='gb18030')

# 2. 划定【不需要】参与统计分析的标识符列
exclude_cols = ['id', 'sample-id', 'group', 'origin-id']

# 3. 明确指定【分类变量】（请根据您的实际需求增删此列表）
categorical_cols = [
    'Female[n(%)]',
    'shock',
    'AKI',
    'INR',
    'ARDS',
    'ARDS-severity',
    'Mechanical ventilation[n(%)]'
]

# 4. 自动推导【连续变量】：除排除列和分类变量外的所有列
numeric_cols = [col for col in df.columns if col not in exclude_cols and col not in categorical_cols]

# 强制将连续变量转换为数值格式，遇到无法识别的字符（如文本、特殊符号）转为空值(NaN)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 获取所有组别名称
groups = df['group'].dropna().unique()
results = []

# ================= 5. 分类变量分析 =================
for col in categorical_cols:
    if col not in df.columns: continue
    valid_data = df[col].dropna()
    if len(valid_data) == 0: continue

    # 整体计算 P 值 (卡方检验)
    try:
        cross_tab = pd.crosstab(df['group'], df[col])
        chi2, p, dof, ex = chi2_contingency(cross_tab)
        p_val_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    except:
        p_val_str = "NA"

    unique_vals = valid_data.unique()
    is_binary = (len(unique_vals) == 2)

    # 寻找二分类的阳性特征 (如果只有 yes/no，就只展示 yes 的占比)
    pos_val = None
    if is_binary:
        lower_vals = [str(x).lower() for x in unique_vals]
        if 'yes' in lower_vals:
            pos_val = unique_vals[lower_vals.index('yes')]
        elif 'female' in lower_vals:
            pos_val = unique_vals[lower_vals.index('female')]
        else:
            pos_val = unique_vals[0]

    # 对每个分类水平进行统计
    for val in unique_vals:
        if is_binary and val != pos_val:
            continue  # 如果是二分类且不是阳性值，则跳过（避免多出一行 no）

        total_count = (df[col] == val).sum()
        total_valid = df[col].notna().sum()

        # 命名格式，例如 ARDS-severity (Mild)
        var_name = col if is_binary else f"{col} ({val})"

        row = {
            'Variable': var_name,
            'Distribution': 'Categorical (Chi-square)',
            f'Total (n={len(df)})': f"{total_count} ({total_count / total_valid * 100:.1f}%)" if total_valid > 0 else "-"
        }

        for g in groups:
            g_data = df[df['group'] == g][col]
            g_count = (g_data == val).sum()
            g_valid = g_data.notna().sum()
            group_name = f"{g} (n={len(df[df['group'] == g])})"
            row[group_name] = f"{g_count} ({g_count / g_valid * 100:.1f}%)" if g_valid > 0 else "-"

        row['P-value'] = p_val_str
        results.append(row)

# ================= 6. 连续变量分析 (自动判定正态性与方差) =================
for col in numeric_cols:
    group_data = [df[df['group'] == g][col].dropna().values for g in groups]
    group_data = [g for g in group_data if len(g) > 0]  # 剔除完全没有数据的组

    if len(group_data) < 2:
        continue  # 只有一组数据或者没有数据，无法比较

    # A. 正态性检验 (Shapiro-Wilk)
    is_normal = True
    for g_vals in group_data:
        if len(g_vals) >= 3:
            stat, p_shap = shapiro(g_vals)
            if p_shap < 0.05:
                is_normal = False
                break
        else:
            is_normal = False

    # B. 方差齐性检验 (Levene)
    is_homoscedastic = False
    if is_normal:
        stat, p_levene = levene(*group_data)
        is_homoscedastic = (p_levene >= 0.05)

    row = {'Variable': col}
    total_data = df[col].dropna()

    # C. 根据数据分布输出不同格式
    if is_normal and is_homoscedastic:
        # 满足正态且方差齐 -> Mean ± SD, ANOVA
        row['Distribution'] = 'Normal (Mean±SD, ANOVA)'
        row[f'Total (n={len(df)})'] = f"{total_data.mean():.1f} ± {total_data.std():.1f}"
        for g in groups:
            g_vals = df[df['group'] == g][col].dropna()
            group_name = f"{g} (n={len(df[df['group'] == g])})"
            row[group_name] = f"{g_vals.mean():.1f} ± {g_vals.std():.1f}" if len(g_vals) > 0 else "-"

        stat, p = f_oneway(*group_data)
        row['P-value'] = f"{p:.3f}" if p >= 0.001 else "<0.001"

    else:
        # 不满足正态或方差不齐 -> Median (Q1-Q3), Kruskal-Wallis H
        row['Distribution'] = 'Non-normal (Median(IQR), Kruskal)'
        row[
            f'Total (n={len(df)})'] = f"{total_data.median():.1f} ({total_data.quantile(0.25):.1f}-{total_data.quantile(0.75):.1f})"
        for g in groups:
            g_vals = df[df['group'] == g][col].dropna()
            group_name = f"{g} (n={len(df[df['group'] == g])})"
            row[group_name] = f"{g_vals.median():.1f} ({g_vals.quantile(0.25):.1f}-{g_vals.quantile(0.75):.1f})" if len(
                g_vals) > 0 else "-"

        stat, p = kruskal(*group_data)
        row['P-value'] = f"{p:.3f}" if p >= 0.001 else "<0.001"

    results.append(row)

# 7. 导出最终结果
result_df = pd.DataFrame(results)

# 组织表格顺序：变量名、分布类型、总体列、各个分组列、P值列
base_cols = ['Variable', 'Distribution', f'Total (n={len(df)})']
group_cols = [f"{g} (n={len(df[df['group'] == g])})" for g in groups]
columns_order = base_cols + group_cols + ['P-value']

result_df = result_df[columns_order]
result_df.to_csv('table1_custom_analysis.csv', index=False, encoding='utf-8-sig')

print("统计分析完成！所有非分类变量已被自动作为连续变量进行分析。")
print("结果已保存为：table1_custom_analysis.csv")