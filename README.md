# A2P3 – An Uncertainty-Aware Collaborative Inference Architecture for Privacy-Preserving On-Device Foundation Models

Author: Uthej Mopathi (author 2, paper 3) – ICE2CPT 2026, Track 7 (smart-city traffic camera analytics across device, edge and cloud).
Proposed method: MAX-MIN Ant Colony Optimisation for uncertainty-aware device–edge–cloud routing and partitioning.
Baselines: cloud-only, device-only, greedy latency, round-robin (+ Greedy-J and UQ-threshold references).

Contents
- data/    synthetic dataset A2P3_traffic_cam_collab_inference.csv + data dictionary (README). The data are SYNTHETIC.
- code/    run_all.py (all experiments, tables, figures), fig1_flowchart.dot, requirements.txt, README.md
- results/ all figures (JPG), all tables (CSV), metrics.json

License: MIT

Reproduce (CPU, ~18 min, Graphviz `dot` required):
    pip install -r code/requirements.txt
    cd code && python3 run_all.py
