from __future__ import annotations

from pathlib import Path

import pandas as pd
import mygene


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR
    / "sepsis_vs_no_sepsis_pydeseq2_results_nonzero20pct"
    / "all_genes_deseq2_results.csv"
)
OUT_DIR = BASE_DIR / "sepsis_vs_no_sepsis_pydeseq2_results_nonzero20pct"
OUT_CSV = OUT_DIR / "all_filtered_genes_formatted_with_description.csv"
OUT_XLSX = OUT_DIR / "all_filtered_genes_formatted_with_description.xlsx"


def fetch_gene_descriptions(symbols: list[str]) -> pd.DataFrame:
    mg = mygene.MyGeneInfo()
    records = mg.querymany(
        symbols,
        scopes="symbol",
        fields="symbol,name",
        species="human",
        as_dataframe=False,
        verbose=False,
    )

    best_hits: dict[str, dict[str, str]] = {}
    for rec in records:
        query = rec.get("query")
        if not query or rec.get("notfound"):
            continue
        current = {
            "query": query,
            "symbol": rec.get("symbol", ""),
            "name": rec.get("name", ""),
            "score": rec.get("_score", 0),
        }
        previous = best_hits.get(query)
        if previous is None:
            best_hits[query] = current
            continue

        current_exact = current["symbol"] == query
        previous_exact = previous["symbol"] == query
        if current_exact and not previous_exact:
            best_hits[query] = current
        elif current_exact == previous_exact and current["score"] > previous["score"]:
            best_hits[query] = current

    return pd.DataFrame(
        {
            "Symbol": symbols,
            "Description": [best_hits.get(symbol, {}).get("name", "") for symbol in symbols],
        }
    )


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["padj", "pvalue"], na_position="last").copy()

    unique_symbols = df["gene"].dropna().astype(str).drop_duplicates().tolist()
    desc_df = fetch_gene_descriptions(unique_symbols)

    out_df = df.rename(
        columns={
            "gene": "Symbol",
            "log2FC": "Log2 Fold Change",
            "pvalue": "p value",
            "padj": "p adj",
        }
    )[["Symbol", "Log2 Fold Change", "p value", "p adj"]]

    out_df = out_df.merge(desc_df, on="Symbol", how="left")
    out_df.to_csv(OUT_CSV, index=False)
    out_df.to_excel(OUT_XLSX, index=False)


if __name__ == "__main__":
    main()
