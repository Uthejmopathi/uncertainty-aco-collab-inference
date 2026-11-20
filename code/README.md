# A2P3 code – uncertainty-aware device–edge–cloud routing with MAX-MIN ACO

Run (Python 3.10+, CPU only, Graphviz `dot` on PATH):

    pip install -r requirements.txt
    python3 run_all.py

`run_all.py` reads `../data/A2P3_traffic_cam_collab_inference.csv` (synthetic) and writes everything to
`../results/` (CSV tables, `metrics.json`, JPG figures) and `../figures/` (300-dpi PNG for the paper).
Seeds are fixed (0–4); single-threaded LightGBM.

Pipeline
1. Pre-processing: time split (2025 = train, 2026 H1 = test), median imputation of the 3 columns with gaps,
   3×IQR clipping of uplink bandwidth and cloud RTT (bounds learned on train only). Outcome columns
   (latency, energy, exposure, error) are never used as features.
2. Outcome models (LightGBM): P(error | context, uncertainty, partition) and log-latency(context, partition, edge load).
   Planner models: train period, one per seed. Digital twin: larger model fit on all data, used only for scoring.
3. Decision windows of 100 consecutive test frames share edge capacity, a cloud quota and a privacy-exposure budget;
   cloud_full is forbidden for privacy_sensitivity >= 1.
4. Policies: Cloud-only, Device-only, Greedy-latency, Round-robin (required baselines), plus Greedy-J (myopic),
   UQ-threshold and the logged random partitions; proposed MAX-MIN ACO with uncertainty-ordered construction,
   congestion-aware cost and 1-flip local search.
5. 5 seeds, Wilcoxon test on per-window cost, ablation (3 seeds), alpha/beta sensitivity, Pareto sweep, privacy study.

Outputs (results/)
- table_main_comparison.csv / main_comparison_per_seed.csv – Table III
- table_significance.csv – Wilcoxon per-window cost tests
- table_ablation.csv / ablation_per_seed.csv – Table IV
- table_sensitivity_alpha_beta.csv, sensitivity_alpha_beta_per_seed.csv – Fig. 7
- pareto_sweep.csv – Fig. 6; aco_partition_mix_by_entropy_quartile.csv – Fig. 5
- privacy_unconstrained_per_seed.csv – ACO without privacy constraints
- table_dataset_summary.csv, table_hyperparameters.csv – Tables I–II
- metrics.json – model quality (AUROC, Brier, R2), convergence summary, all aggregates
- fig1_flowchart.(jpg|dot) … fig7_alpha_beta.jpg
