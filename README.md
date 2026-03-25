# validation

CPM > 1 validation materials for the cfRNA delivery analysis.

Contents:
- `cpm_gt1_validation/analyze_cpm_gt1_groups.py`: analysis script
- `cpm_gt1_validation/sample_cpm_gt1_gene_counts_by_group.csv`: per-sample counts
- `cpm_gt1_validation/sample_cpm_gt1_gene_counts_stats.csv`: group-level statistics
- `cpm_gt1_validation/sample_cpm_gt1_gene_counts_barplot.svg`: exported figure
- `sepsis_deseq2_validation/deseq2_sepsis_vs_no_sepsis.py`: PyDESeq2 comparison for sepsis vs no-sepsis
- `sepsis_deseq2_validation/export_all_filtered_genes_with_descriptions.py`: export filtered DE genes with gene descriptions

Note:
- The script currently uses absolute local paths from the original workstation.
- Large source matrices are not included in this repository snapshot.
