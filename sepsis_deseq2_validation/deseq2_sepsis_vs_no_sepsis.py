from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


BASE_DIR = Path(__file__).resolve().parent
GROUP_FILE = BASE_DIR / "group.csv"
EXP_FILE = BASE_DIR / "exp.csv"
OUT_DIR = BASE_DIR / "sepsis_vs_no_sepsis_pydeseq2_results_nonzero20pct"

TARGET_GROUPS = ["sepsis", "No-sepsis"]
REF_GROUP = "No-sepsis"
TEST_GROUP = "sepsis"
MIN_NONZERO_FRACTION = 0.20
ALPHA = 0.05
LOG2FC_THRESHOLD = 1.0


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    raw_group_df = pd.read_csv(GROUP_FILE).rename(columns={"nsmple": "sample"})
    group_df = raw_group_df[raw_group_df["group"].isin(TARGET_GROUPS)].copy()
    group_df["group"] = pd.Categorical(group_df["group"], categories=[REF_GROUP, TEST_GROUP], ordered=True)
    group_df = group_df.sort_values(["group", "sample"]).reset_index(drop=True)

    exp_df = pd.read_csv(EXP_FILE).rename(columns={"Unnamed: 0": "gene"})
    exp_df = exp_df[["gene", *group_df["sample"].tolist()]].copy()
    exp_df = exp_df.drop_duplicates(subset="gene", keep="first").set_index("gene")

    counts = exp_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    min_samples = max(1, int(len(group_df) * MIN_NONZERO_FRACTION))
    keep_mask = (counts > 0).sum(axis=1) >= min_samples
    counts = counts.loc[keep_mask].copy()

    counts_for_deseq = counts.T.copy()
    metadata = group_df.rename(columns={"group": "condition"}).set_index("sample").loc[counts_for_deseq.index]
    metadata["condition"] = metadata["condition"].astype(str)

    dds = DeseqDataSet(
        counts=counts_for_deseq,
        metadata=metadata,
        design="~condition",
        n_cpus=1,
        quiet=True,
    )
    dds.deseq2()

    stat_res = DeseqStats(
        dds,
        contrast=["condition", TEST_GROUP, REF_GROUP],
        alpha=ALPHA,
        quiet=True,
        n_cpus=1,
    )
    stat_res.summary()

    coeff_candidates = [col for col in stat_res.LFC.columns if "condition" in str(col)]
    shrink_coeff = coeff_candidates[0] if coeff_candidates else stat_res.LFC.columns[-1]
    stat_res.lfc_shrink(coeff=shrink_coeff)

    results = stat_res.results_df.copy()
    results.index.name = "gene"
    results = results.rename(
        columns={
            "log2FoldChange": "log2FC",
            "lfcSE": "lfcSE",
            "stat": "stat",
            "pvalue": "pvalue",
            "padj": "padj",
            "baseMean": "baseMean",
        }
    )
    results["direction"] = results["log2FC"].apply(
        lambda x: "up_in_sepsis" if pd.notna(x) and x > 0 else "up_in_no_sepsis"
    )
    results["significant"] = (
        results["padj"].notna()
        & (results["padj"] < ALPHA)
        & (results["log2FC"].abs() >= LOG2FC_THRESHOLD)
    )
    results = results.sort_values(["padj", "pvalue"], na_position="last")

    sig_results = results[results["significant"]].copy()
    up_results = sig_results[sig_results["log2FC"] >= LOG2FC_THRESHOLD].copy()
    down_results = sig_results[sig_results["log2FC"] <= -LOG2FC_THRESHOLD].copy()
    top50_results = results.head(50).copy()

    results.reset_index().to_csv(OUT_DIR / "all_genes_deseq2_results.csv", index=False)
    sig_results.reset_index().to_csv(OUT_DIR / "significant_genes_fdr0.05_log2fc1.csv", index=False)
    up_results.reset_index().to_csv(OUT_DIR / "up_in_sepsis_genes.csv", index=False)
    down_results.reset_index().to_csv(OUT_DIR / "up_in_no_sepsis_genes.csv", index=False)
    top50_results.reset_index().to_csv(OUT_DIR / "top50_candidate_genes.csv", index=False)
    metadata.reset_index().to_csv(OUT_DIR / "analysis_samples.csv", index=False)
    raw_group_df[~raw_group_df["group"].isin(TARGET_GROUPS)].to_csv(
        OUT_DIR / "excluded_non_target_samples.csv", index=False
    )

    summary_lines = [
        "Comparison: sepsis vs No-sepsis",
        f"Input expression file: {EXP_FILE}",
        f"Input group file: {GROUP_FILE}",
        f"Total samples in group table: {len(raw_group_df)}",
        f"Samples retained for analysis: {len(metadata)}",
        f"sepsis samples: {(metadata['condition'] == TEST_GROUP).sum()}",
        f"No-sepsis samples: {(metadata['condition'] == REF_GROUP).sum()}",
        "Excluded groups: "
        + ", ".join(
            f"{group}={count}"
            for group, count in raw_group_df.loc[
                ~raw_group_df["group"].isin(TARGET_GROUPS), "group"
            ].value_counts().items()
        ),
        f"Genes before filtering: {exp_df.shape[0]}",
        f"Genes after filter (non-zero counts in at least {min_samples} samples; {MIN_NONZERO_FRACTION:.0%} of samples): {counts.shape[0]}",
        "Method: PyDESeq2 negative binomial model + Wald test + BH-adjusted p-values",
        f"Reference level: {REF_GROUP}",
        f"Contrast tested: {TEST_GROUP} vs {REF_GROUP}",
        f"Shrinkage coefficient: {shrink_coeff}",
        f"Significance rule: padj < {ALPHA} and |log2FC| >= {LOG2FC_THRESHOLD}",
        f"Significant genes: {sig_results.shape[0]}",
        f"Up in sepsis: {up_results.shape[0]}",
        f"Up in No-sepsis: {down_results.shape[0]}",
        "Top candidate genes by padj: " + ", ".join(top50_results.head(10).index.tolist()),
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
