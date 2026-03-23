import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from scipy import stats


BASE = Path(r"D:\athesis-data\human_cfrna_delivery_report\04.Expression\01_TPM_STAT\combine_readcount")
CPM_PATH = BASE / "combined_geneSymbol_mRNA_CPM.csv"
SAMPLE_INFO_PATH = BASE / "sample_info.csv"
COUNTS_OUT = BASE / "sample_cpm_gt1_gene_counts_by_group.csv"
STATS_OUT = BASE / "sample_cpm_gt1_gene_counts_stats.csv"
PLOT_OUT = BASE / "sample_cpm_gt1_gene_counts_barplot.svg"


def parse_misnamed_xlsx_table(path: Path) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared_strings = []
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in shared_root.findall("a:si", ns):
            shared_strings.append("".join(t.text or "" for t in si.iterfind(".//a:t", ns)))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            values = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                col = re.sub(r"\d+", "", ref)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", ns)
                if value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[col] = value
            rows.append(values)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    ordered_cols = sorted(rows[0].keys(), key=lambda s: (len(s), s))
    headers = [rows[0][col] for col in ordered_cols]
    records = []
    for row in rows[1:]:
        record = {}
        for col, header in zip(ordered_cols, headers):
            record[header] = row.get(col, "")
        records.append(record)

    df = pd.DataFrame(records)
    return df[(df["sample"] != "") & (df["group"] != "")].copy()


def map_samples(sample_info: pd.DataFrame, cpm_columns: list[str]) -> pd.DataFrame:
    mapped = sample_info.copy()
    cpm_set = set(cpm_columns)
    resolved = []
    for sample in mapped["sample"]:
        if sample in cpm_set:
            resolved.append(sample)
            continue
        matches = [col for col in cpm_columns if col.endswith(sample)]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely map sample {sample!r}; matches={matches}")
        resolved.append(matches[0])
    mapped["cpm_sample"] = resolved
    return mapped


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        value = min(prev, pvalues[idx] * n / i)
        adjusted[idx] = value
        prev = value
    return adjusted


def p_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return f"p={p:.3f}"


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_svg(summary: pd.DataFrame, pairwise: pd.DataFrame, overall_p: float, out_path: Path) -> None:
    width = 1100
    height = 760
    margin_left = 110
    margin_right = 60
    margin_top = 110
    margin_bottom = 140
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_point = max(max(values) for values in summary["count"])
    max_bar = float((summary["mean"] + summary["sd"]).max())
    y_max = max(max_point, max_bar) * 1.28
    y_max = max(y_max, 1000.0)

    def y_to_px(y: float) -> float:
        return margin_top + plot_height - (y / y_max) * plot_height

    group_names = summary["group"].tolist()
    n_groups = len(group_names)
    colors = ["#5b8ff9", "#61dDAa", "#f6bd16"]
    centers = [margin_left + plot_width * (i + 0.5) / n_groups for i in range(n_groups)]
    bar_width = plot_width / (n_groups * 2.6)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="42" text-anchor="middle" font-size="28" font-family="Arial" font-weight="bold">Genes with CPM &gt; 1 Across Three Groups</text>',
        f'<text x="550" y="74" text-anchor="middle" font-size="16" font-family="Arial">Overall Kruskal-Wallis p = {overall_p:.4g}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="2"/>',
    ]

    for tick in range(6):
        y_val = y_max * tick / 5
        y = y_to_px(y_val)
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="14" font-family="Arial">{int(round(y_val))}</text>')

    lines.append(
        f'<text x="30" y="{margin_top + plot_height / 2}" transform="rotate(-90 30 {margin_top + plot_height / 2})" '
        'text-anchor="middle" font-size="18" font-family="Arial">Number of genes with CPM &gt; 1</text>'
    )

    for i, (_, row) in enumerate(summary.iterrows()):
        center = centers[i]
        top = y_to_px(float(row["mean"]))
        bottom = y_to_px(0)
        bar_height = bottom - top
        color = colors[i % len(colors)]
        lines.append(
            f'<rect x="{center - bar_width / 2:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'fill="{color}" fill-opacity="0.85" stroke="#333" stroke-width="1.2"/>'
        )
        err_top = y_to_px(float(row["mean"] + row["sd"]))
        err_bottom = y_to_px(max(float(row["mean"] - row["sd"]), 0.0))
        lines.append(f'<line x1="{center:.1f}" y1="{err_top:.1f}" x2="{center:.1f}" y2="{err_bottom:.1f}" stroke="#222" stroke-width="2"/>')
        lines.append(f'<line x1="{center - 18:.1f}" y1="{err_top:.1f}" x2="{center + 18:.1f}" y2="{err_top:.1f}" stroke="#222" stroke-width="2"/>')
        lines.append(f'<line x1="{center - 18:.1f}" y1="{err_bottom:.1f}" x2="{center + 18:.1f}" y2="{err_bottom:.1f}" stroke="#222" stroke-width="2"/>')

        values = row["count"]
        n = len(values)
        for j, value in enumerate(values):
            jitter = 0 if n == 1 else (-bar_width * 0.28) + (j / (n - 1)) * (bar_width * 0.56)
            px = center + jitter
            py = y_to_px(float(value))
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.8" fill="#222" fill-opacity="0.8"/>')

        lines.append(f'<text x="{center:.1f}" y="{margin_top + plot_height + 32}" text-anchor="middle" font-size="16" font-family="Arial">{svg_escape(row["group"])}</text>')
        lines.append(f'<text x="{center:.1f}" y="{margin_top + plot_height + 56}" text-anchor="middle" font-size="13" font-family="Arial" fill="#555">n={int(row["n"])}</text>')

    pairwise_levels = [y_max * 0.89, y_max * 0.95, y_max * 1.01]
    group_to_x = dict(zip(group_names, centers))
    for level, (_, row) in zip(pairwise_levels, pairwise.iterrows()):
        x1 = group_to_x[row["group1"]]
        x2 = group_to_x[row["group2"]]
        y = y_to_px(level)
        lines.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y + 14:.1f}" stroke="#111" stroke-width="1.8"/>')
        lines.append(f'<line x1="{x2:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y + 14:.1f}" stroke="#111" stroke-width="1.8"/>')
        lines.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="#111" stroke-width="1.8"/>')
        lines.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="15" font-family="Arial">{svg_escape(p_label(float(row["p_adj"])))}</text>')

    legend_y = height - 48
    for i, group in enumerate(group_names):
        x = margin_left + i * 240
        color = colors[i % len(colors)]
        lines.append(f'<rect x="{x}" y="{legend_y - 14}" width="18" height="18" fill="{color}" fill-opacity="0.85" stroke="#333" stroke-width="1"/>')
        lines.append(f'<text x="{x + 28}" y="{legend_y}" font-size="14" font-family="Arial">{svg_escape(group)}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cpm = pd.read_csv(CPM_PATH)
    sample_cols = [col for col in cpm.columns if col != "Symbol"]

    sample_info = parse_misnamed_xlsx_table(SAMPLE_INFO_PATH)
    sample_info = map_samples(sample_info, sample_cols)

    count_series = (cpm[sample_cols] > 1).sum(axis=0)
    counts = sample_info[["sample", "group", "batch", "cpm_sample"]].copy()
    counts["cpm_gt1_gene_count"] = counts["cpm_sample"].map(count_series)
    counts = counts.sort_values(["group", "batch", "sample"]).reset_index(drop=True)
    counts.to_csv(COUNTS_OUT, index=False)

    grouped = {
        group: sub["cpm_gt1_gene_count"].tolist()
        for group, sub in counts.groupby("group", sort=False)
    }
    overall_stat, overall_p = stats.kruskal(*grouped.values())

    group_order = list(grouped.keys())
    pairwise_rows = []
    raw_pvalues = []
    for i in range(len(group_order)):
        for j in range(i + 1, len(group_order)):
            g1 = group_order[i]
            g2 = group_order[j]
            stat, pvalue = stats.mannwhitneyu(grouped[g1], grouped[g2], alternative="two-sided")
            raw_pvalues.append(pvalue)
            pairwise_rows.append(
                {
                    "test": "Mann-Whitney U",
                    "group1": g1,
                    "group2": g2,
                    "statistic": stat,
                    "p_value": pvalue,
                }
            )
    adjusted = benjamini_hochberg(raw_pvalues)
    for row, p_adj in zip(pairwise_rows, adjusted):
        row["p_adj"] = p_adj

    summary_rows = []
    for group in group_order:
        sub = counts.loc[counts["group"] == group, "cpm_gt1_gene_count"]
        summary_rows.append(
            {
                "group": group,
                "n": int(sub.shape[0]),
                "mean": sub.mean(),
                "sd": sub.std(ddof=1),
                "median": sub.median(),
                "min": sub.min(),
                "max": sub.max(),
                "count": sub.tolist(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    stats_rows = [
        {
            "test": "Kruskal-Wallis",
            "group1": "ALL",
            "group2": "ALL",
            "statistic": overall_stat,
            "p_value": overall_p,
            "p_adj": overall_p,
        }
    ] + pairwise_rows
    pd.DataFrame(stats_rows).to_csv(STATS_OUT, index=False)

    build_svg(summary, pd.DataFrame(pairwise_rows), overall_p, PLOT_OUT)

    print("sample counts by group")
    print(counts.groupby("group")["sample"].count().to_string())
    print("\nsummary")
    print(summary[["group", "n", "mean", "sd", "median", "min", "max"]].to_string(index=False))
    print("\nstatistics")
    print(pd.DataFrame(stats_rows).to_string(index=False))
    print("\noutputs")
    print(COUNTS_OUT)
    print(STATS_OUT)
    print(PLOT_OUT)


if __name__ == "__main__":
    main()
