#!/usr/bin/env python3
"""A2P3 - Uncertainty-aware collaborative device-edge-cloud inference for traffic cameras.

Proposed method: MAX-MIN Ant Colony Optimisation (ACO) that jointly routes / partitions every frame of a
decision window (device_only, split_after_block2, split_after_block4, edge_full, cloud_full) under shared edge
capacity, a cloud quota, a privacy-exposure budget and a hard privacy rule, using uncertainty-aware outcome models.
Baselines: cloud-only, device-only, greedy latency, round-robin (+ myopic greedy-J, uncertainty threshold, logged).

Runs everything: preprocessing, outcome models, digital twin, all policies over 5 seeds, ablation, sensitivity,
Pareto sweep, significance tests, all figures and tables.  Usage:  python3 run_all.py
"""
import os, json, time, subprocess, shutil, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, wilcoxon
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, f1_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "A2P3_traffic_cam_collab_inference.csv")
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
T0 = time.time()
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 100})
ACC = "#c0392b"; GREY = "#7f8c8d"

ACTIONS = ["device_only", "split_after_block2", "split_after_block4", "edge_full", "cloud_full"]
SHORT = ["Dev", "Split2", "Split4", "Edge", "Cloud"]
NA = len(ACTIONS)
EDGE_UNITS = np.array([0.0, 0.5, 0.3, 1.0, 0.0])       # edge compute units consumed per frame
IS_CLOUD = np.array([0, 0, 0, 0, 1])
SEEDS = [0, 1, 2, 3, 4]
# ---------------- problem constants (decision window) ----------------
B = 100            # frames per decision window
K_EDGE = 25.0      # edge capacity units per window
Q_CLOUD = 15       # cloud quota (frames per window)
EPS_EXP = 0.05     # mean privacy-exposure budget per window
KAPPA = 0.6        # congestion: effective edge load = load + KAPPA * u
UGRID = np.linspace(0, 1, 11)
DEADLINE = {"incident_detection": 250.0, "pedestrian_safety": 250.0, "vehicle_count": 500.0, "plate_privacy_blur": 500.0}
W = dict(err=1.0, lat=0.10, miss=0.50, en=0.05)   # cost weights: per frame; latency per 100 ms, energy per J
ACO_P = dict(ants=24, iters=50, alpha=1.0, beta=2.0, rho=0.10, stall=15, ls_every=10)

# ======================= 1. load + preprocess =======================
df = pd.read_csv(DATA, parse_dates=["frame_ts"]).sort_values("frame_ts").reset_index(drop=True)
n_missing = df.isna().sum()
train = df[df.frame_ts < "2026-01-01"].copy(); test = df[df.frame_ts >= "2026-01-01"].copy()
miss_cols = ["device_mc_dropout_var", "uplink_bandwidth_mbps", "illuminance_lux"]
med = {c: train[c].median() for c in miss_cols}
clipq = {}
for c in ["uplink_bandwidth_mbps", "cloud_rtt_ms"]:
    q1, q3 = train[c].quantile([0.25, 0.75]); iqr = q3 - q1
    clipq[c] = (max(train[c].min(), q1 - 3 * iqr), q3 + 3 * iqr)
n_out = {c: int(((df[c] < clipq[c][0]) | (df[c] > clipq[c][1])).sum()) for c in clipq}

def prep(d):
    d = d.copy()
    for c in miss_cols: d[c] = d[c].fillna(med[c])
    for c, (lo, hi) in clipq.items(): d[c] = d[c].clip(lo, hi)
    d["log_lux"] = np.log1p(d.illuminance_lux); d["log_bw"] = np.log(d.uplink_bandwidth_mbps)
    d["hour"] = d.frame_ts.dt.hour + d.frame_ts.dt.minute / 60
    return d
df = prep(df); train = df[df.frame_ts < "2026-01-01"].copy(); test = df[df.frame_ts >= "2026-01-01"].copy()

CAT = {"camera_site_type": ["intersection", "highway", "pedestrian_crossing", "bus_lane"],
       "analytics_task": list(DEADLINE), "weather": ["clear", "rain", "fog", "snow"]}
UQ = ["device_pred_entropy", "device_mc_dropout_var", "device_top1_confidence"]
NUM_CTX = ["log_lux", "night", "objects_in_frame", "occlusion_ratio", "privacy_sensitivity", "log_bw",
           "edge_node_load", "cloud_rtt_ms", "device_battery_pct", "hour"]

def feats(d, action_idx, use_uq=True, load_override=None):
    X = pd.DataFrame(index=d.index)
    for c in NUM_CTX + (UQ if use_uq else []): X[c] = d[c].values
    if load_override is not None: X["edge_node_load"] = load_override
    for c, lv in CAT.items():
        for v in lv: X[f"{c}={v}"] = (d[c].values == v).astype(float)
    a = np.broadcast_to(np.asarray(action_idx), (len(d),))
    for k, an in enumerate(ACTIONS): X[f"a={an}"] = (a == k).astype(float)
    X["a_cap"] = np.array([0, 1, 1, 2, 3])[a]
    return X

aidx = lambda d: d.partition_point.map({a: i for i, a in enumerate(ACTIONS)}).values

# ======================= 2. outcome models =======================
def fit_models(d, seed, use_uq=True, big=False):
    Xa = feats(d, aidx(d), use_uq)
    par = dict(n_estimators=600 if big else 300, learning_rate=0.03 if big else 0.05, num_leaves=31 if big else 15,
               min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
               num_threads=1, verbose=-1)
    clf = lgb.LGBMClassifier(**par).fit(Xa, d.prediction_error.values)
    reg = lgb.LGBMRegressor(**par).fit(Xa, np.log(d.latency_ms.values))
    resid = np.log(d.latency_ms.values) - reg.predict(Xa)
    sig = np.array([resid[aidx(d) == k].std() for k in range(NA)])
    energy = d.groupby("partition_point").device_energy_j.mean().reindex(ACTIONS).values
    expo = d.groupby(["partition_point", "privacy_sensitivity"]).privacy_exposure.mean().unstack().reindex(ACTIONS).fillna(0).values
    return dict(clf=clf, reg=reg, sig=sig, energy=energy, expo=expo, use_uq=use_uq)

def outcome_tensors(m, d):
    """per frame x action: p_err, latency mu over load grid (edge actions), energy, exposure."""
    n = len(d); P = np.zeros((n, NA)); MU = np.zeros((n, NA, len(UGRID)))
    for k in range(NA):
        P[:, k] = m["clf"].predict_proba(feats(d, k, m["use_uq"]))[:, 1]
        if EDGE_UNITS[k] > 0:
            for g, u in enumerate(UGRID):
                MU[:, k, g] = m["reg"].predict(feats(d, k, m["use_uq"], np.clip(d.edge_node_load.values + KAPPA * u, 0, 1)))
        else:
            MU[:, k, :] = m["reg"].predict(feats(d, k, m["use_uq"]))[:, None]
    EL = np.exp(MU + m["sig"][None, :, None] ** 2 / 2)
    D = d.analytics_task.map(DEADLINE).values
    PM = 1 - norm.cdf((np.log(D)[:, None, None] - MU) / m["sig"][None, :, None])
    E = np.tile(m["energy"], (n, 1)); X = m["expo"][:, d.privacy_sensitivity.values].T
    return dict(P=P, EL=EL, PM=PM, E=E, X=X)

def cost_grid(T, d, w=W):
    lowb = (d.device_battery_pct.values < 30).astype(float)
    wen = w["en"] * (1 + 2 * lowb)
    return (w["err"] * T["P"][:, :, None] + w["lat"] * T["EL"] / 100.0 + w["miss"] * T["PM"]
            + (wen[:, None] * T["E"])[:, :, None])

# planner (train period) models, twin (all data, larger) models
print("fitting models ...")
twin = fit_models(df, 123, True, big=True)
TW = outcome_tensors(twin, test)
planners = {s: fit_models(train, s, True) for s in SEEDS}
plan_noUQ = fit_models(train, 0, False)

# model quality on test (logged actions)
Xt = feats(test, aidx(test)); yt = test.prediction_error.values
pm0 = planners[0]["clf"].predict_proba(Xt)[:, 1]
pm_nouq = plan_noUQ["clf"].predict_proba(feats(test, aidx(test), False))[:, 1]
ent = test.device_pred_entropy.values
# uncertainty-only model: entropy, var, conf + action
Xu_tr = pd.concat([train[UQ].reset_index(drop=True), pd.get_dummies(pd.Categorical(train.partition_point, ACTIONS)).astype(float).reset_index(drop=True)], axis=1)
Xu_te = pd.concat([test[UQ].reset_index(drop=True), pd.get_dummies(pd.Categorical(test.partition_point, ACTIONS)).astype(float).reset_index(drop=True)], axis=1)
Xu_tr.columns = Xu_te.columns = [str(c) for c in Xu_tr.columns]
uq_only = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=0, num_threads=1, verbose=-1).fit(Xu_tr, train.prediction_error.values)
pm_uq = uq_only.predict_proba(Xu_te)[:, 1]
dev_mask = aidx(test) == 0
model_q = {
    "auroc_full": roc_auc_score(yt, pm0), "auroc_no_uq": roc_auc_score(yt, pm_nouq), "auroc_uq_only": roc_auc_score(yt, pm_uq),
    "auroc_entropy_raw_device_only": roc_auc_score(yt[dev_mask], ent[dev_mask]),
    "f1_full@0.5": f1_score(yt, pm0 > 0.5), "f1_no_uq@0.5": f1_score(yt, pm_nouq > 0.5),
    "brier_full": brier_score_loss(yt, pm0), "brier_no_uq": brier_score_loss(yt, pm_nouq),
    "latency_r2_log": r2_score(np.log(test.latency_ms.values), planners[0]["reg"].predict(Xt)),
    "auroc_full_seeds_mean": float(np.mean([roc_auc_score(yt, planners[s]["clf"].predict_proba(Xt)[:, 1]) for s in SEEDS])),
    "auroc_full_seeds_std": float(np.std([roc_auc_score(yt, planners[s]["clf"].predict_proba(Xt)[:, 1]) for s in SEEDS])),
}
model_q = {k: float(v) for k, v in model_q.items()}
print(model_q)

# ======================= 3. windows + policies =======================
starts = list(range(0, len(test), B)); windows = [np.arange(s, min(s + B, len(test))) for s in starts]
PRIV = test.privacy_sensitivity.values; TASK = test.analytics_task.values

def feasible_static(i):
    m = np.ones(NA, bool)
    if PRIV[i] >= 1: m[4] = False          # hard rule: no raw frame to cloud if faces/plates in scope
    return m

def solution_cost(G, A, units):
    """G: (b,NA,11) cost grid, A: (n_sol,b) actions -> total cost per solution (with congestion u)."""
    u = np.clip(units[A].sum(1) / K_EDGE, 0, 1)
    pos = u * (len(UGRID) - 1); k0 = np.floor(pos).astype(int).clip(0, len(UGRID) - 2); f = pos - k0
    rows = np.arange(A.shape[1])[None, :]
    c0 = G[rows, A, k0[:, None]]; c1 = G[rows, A, k0[:, None] + 1]
    return (c0 * (1 - f[:, None]) + c1 * f[:, None]).sum(1)

class Budget:
    def __init__(self, Xexp, b):
        self.edge = K_EDGE; self.cloud = Q_CLOUD; self.exp = EPS_EXP * b; self.Xexp = Xexp
    def ok(self, i, k):
        return (EDGE_UNITS[k] <= self.edge + 1e-9) and (IS_CLOUD[k] <= self.cloud) and (self.Xexp[i, k] <= self.exp + 1e-9)
    def take(self, i, k):
        self.edge -= EDGE_UNITS[k]; self.cloud -= IS_CLOUD[k]; self.exp -= self.Xexp[i, k]

def seq_policy(idx, Tpl, score_fn, order=None):
    """sequential feasible assignment: each frame picks argmin score among feasible actions."""
    b = len(idx); bud = Budget(Tpl["X"][idx], b); A = np.zeros(b, int)
    for j in (order if order is not None else range(b)):
        i = idx[j]; sc = score_fn(j).astype(float).copy()
        for k in range(NA):
            if not (feasible_static(i)[k] and bud.ok(j, k)): sc[k] = np.inf
        A[j] = int(np.argmin(sc)); bud.take(j, A[j])
    return A

def fixed_policy(idx, Tpl, prefs):
    """try actions in preference order, fall back when infeasible (device always feasible)."""
    return seq_policy(idx, Tpl, lambda j: np.array([prefs.index(k) if k in prefs else 99 for k in range(NA)]))

def round_robin(idx, Tpl, offset):
    cyc = [0, 1, 2, 3, 4]; b = len(idx); bud = Budget(Tpl["X"][idx], b); A = np.zeros(b, int)
    for j in range(b):
        i = idx[j]
        for t in range(NA):
            k = cyc[(offset + j + t) % NA]
            if feasible_static(i)[k] and bud.ok(j, k): break
        else: k = 0
        A[j] = k; bud.take(j, k)
    return A

def aco(idx, Tpl, G, rng, p=ACO_P, use_heur=True, use_ls=True, congestion=True, order_mode="uq", trace=None):
    b = len(idx); m = p["ants"]; Xexp = Tpl["X"][idx]
    Gp = G if congestion else np.repeat(G[:, :, :1], len(UGRID), axis=2)
    stat = np.array([feasible_static(i) for i in idx])
    order = np.argsort(-Tpl["P"][idx, 0]) if order_mode == "uq" else np.arange(b)
    # warm start: myopic greedy cost for tau bounds
    A_g = seq_policy(idx, Tpl, lambda j: Gp[j, :, 0], order)
    best_A = A_g.copy(); best_c = solution_cost(G, best_A[None], EDGE_UNITS)[0]
    tmax = 1.0 / (p["rho"] * best_c); tmin = tmax / (2 * b)
    tau = np.full((b, NA), tmax); stall = 0
    for it in range(p["iters"]):
        A = np.zeros((m, b), int)
        ed = np.full(m, K_EDGE); cl = np.full(m, float(Q_CLOUD)); ex = np.full(m, EPS_EXP * b)
        for j in order:
            u = np.clip((K_EDGE - ed) / K_EDGE, 0, 1); g = np.rint(u * (len(UGRID) - 1)).astype(int)
            c = Gp[j][:, g].T                                    # (m, NA) marginal cost at current load
            feas = stat[j][None, :] & (EDGE_UNITS[None, :] <= ed[:, None] + 1e-9) & \
                   (IS_CLOUD[None, :] <= cl[:, None]) & (Xexp[j][None, :] <= ex[:, None] + 1e-9)
            eta = 1.0 / (c + 1e-3) if use_heur else np.ones_like(c)
            w = (tau[j][None, :] ** p["alpha"]) * (eta ** p["beta"]) * feas
            w = w / w.sum(1, keepdims=True)
            k = (w.cumsum(1) > rng.random((m, 1))).argmax(1)
            A[:, j] = k; ed -= EDGE_UNITS[k]; cl -= IS_CLOUD[k]; ex -= Xexp[j, k]
        costs = solution_cost(Gp, A, EDGE_UNITS)
        ib = int(np.argmin(costs)); A_ib = A[ib].copy()
        if use_ls and ((it + 1) % p["ls_every"] == 0 or it == p["iters"] - 1):
            A_ib = local_search(A_ib, Gp, stat, Xexp)
        c_ib = solution_cost(Gp, A_ib[None], EDGE_UNITS)[0]
        if c_ib < best_c - 1e-9: best_c, best_A, stall = c_ib, A_ib.copy(), 0
        else: stall += 1
        tmax = 1.0 / (p["rho"] * best_c); tmin = tmax / (2 * b)
        tau *= (1 - p["rho"])
        dep = best_A if it % 2 else A_ib                      # alternate iteration-best / global-best deposit
        tau[np.arange(b), dep] += 1.0 / solution_cost(Gp, dep[None], EDGE_UNITS)[0]
        tau = np.clip(tau, tmin, tmax)
        if trace is not None: trace.append(best_c)
        if stall >= p["stall"]: break
    if trace is not None:
        trace += [best_c] * (p["iters"] - len(trace))
    return best_A

def local_search(A, G, stat, Xexp):
    A = A.copy(); b = len(A); improved = True; cur = solution_cost(G, A[None], EDGE_UNITS)[0]
    while improved:
        improved = False
        cand = np.repeat(A[None], b * NA, 0)
        jj = np.repeat(np.arange(b), NA); kk = np.tile(np.arange(NA), b); cand[np.arange(b * NA), jj] = kk
        ok = stat[jj, kk] & (EDGE_UNITS[cand].sum(1) <= K_EDGE + 1e-9) & (IS_CLOUD[cand].sum(1) <= Q_CLOUD) & \
             (Xexp[np.arange(b)[None, :], cand].sum(1) <= EPS_EXP * b + 1e-9)
        cc = solution_cost(G, cand, EDGE_UNITS); cc[~ok] = np.inf
        q = int(np.argmin(cc))
        if cc[q] < cur - 1e-9: A, cur, improved = cand[q].copy(), cc[q], True
    return A

# ======================= 4. evaluation via twin =======================
TWG = cost_grid(TW, test)
def evaluate(A_full):
    """A_full: action per test frame. Twin outcomes with congestion u per window."""
    p_err = np.zeros(len(test)); lat = np.zeros(len(test)); miss = np.zeros(len(test)); J = []
    for idx in windows:
        A = A_full[idx]; u = np.clip(EDGE_UNITS[A].sum() / K_EDGE, 0, 1); pos = u * 10
        k0 = min(int(np.floor(pos)), 9); f = pos - k0
        r = np.arange(len(idx))
        p_err[idx] = TW["P"][idx, A]
        lat[idx] = TW["EL"][idx, A, k0] * (1 - f) + TW["EL"][idx, A, k0 + 1] * f
        miss[idx] = TW["PM"][idx, A, k0] * (1 - f) + TW["PM"][idx, A, k0 + 1] * f
        J.append(solution_cost(TWG[idx], A[None], EDGE_UNITS)[0] / len(idx))
    en = TW["E"][np.arange(len(test)), A_full]; ex = TW["X"][np.arange(len(test)), A_full]
    I = np.concatenate(windows)                       # frames covered by the evaluated windows
    # model-free IPS estimate of accuracy using the (near-uniform) logged random partitions
    prop = train.partition_point.value_counts(normalize=True).reindex(ACTIONS).values
    la = aidx(test); match = (la == A_full)
    ips = float(np.mean((match * yt / prop[la])[I]) / np.mean((match / prop[la])[I]))   # self-normalised IPS
    win_exp = np.array([ex[i].mean() for i in windows])
    return {"accuracy_pct": 100 * (1 - p_err[I].mean()), "mean_latency_ms": lat[I].mean(),
            "p95_latency_ms": float(np.percentile(lat[I], 95)), "deadline_miss_pct": 100 * miss[I].mean(),
            "energy_j": en[I].mean(), "exposure": ex[I].mean(), "max_window_exposure": win_exp.max(),
            "priv_violations": int(((A_full[I] == 4) & (PRIV[I] >= 1)).sum()), "J_per_frame": float(np.mean(J)),
            "ips_accuracy_pct": 100 * (1 - ips), "offload_pct": 100 * (A_full[I] != 0).mean()}, np.array(J)

def run_policy(name, seed, planner=None, **kw):
    rng = np.random.default_rng(seed); pl = planner or planners[seed]
    Tpl = TPL[id(pl)]; G = GPL[id(pl)]
    A_full = np.zeros(len(test), int); t0 = time.time(); traces = []
    for w_i, idx in enumerate(windows):
        if name == "Cloud-only": A = fixed_policy(idx, Tpl, [4, 3, 2, 1, 0])
        elif name == "Device-only": A = np.zeros(len(idx), int)
        elif name == "Greedy-latency": A = seq_policy(idx, Tpl, lambda j: Tpl["EL"][idx[j], :, 0] + np.array([1e6, 0, 0, 0, 0]))  # fastest offload tier
        elif name == "Round-robin": A = round_robin(idx, Tpl, int(rng.integers(NA)))
        elif name == "Greedy-J (myopic)": A = seq_policy(idx, Tpl, lambda j: G[idx[j], :, 0])
        elif name == "UQ-threshold": A = uq_threshold(idx, Tpl)
        elif name == "Logged (random)": A = aidx(test)[idx]
        else:
            tr = [] if kw.get("trace") else None
            A = aco(idx, Tpl, G[idx], rng, **{k: v for k, v in kw.items() if k != "trace"}, trace=tr)
            if tr is not None: traces.append(tr)
        A_full[idx] = A
    met, J = evaluate(A_full)
    if name == "Logged (random)": met["ips_accuracy_pct"] = 100 * (1 - yt[np.concatenate(windows)].mean())   # on-policy
    met["runtime_ms_per_window"] = 1000 * (time.time() - t0) / len(windows)
    return met, J, A_full, traces

def uq_threshold(idx, Tpl, tau=0.35):
    """offload to edge_full when predicted device error > tau (else device); feasibility fallback."""
    return seq_policy(idx, Tpl, lambda j: np.array([9, 2, 1, 0, 3]) if Tpl["P"][idx[j], 0] > tau else np.array([0, 9, 9, 9, 9]))

TPL, GPL = {}, {}
def register(pl, w=W):
    T = outcome_tensors(pl, test); TPL[id(pl)] = T; GPL[id(pl)] = cost_grid(T, test, w)
for s in SEEDS: register(planners[s])
register(plan_noUQ)
print("tensors ready %.0fs" % (time.time() - T0))

METHODS = ["Cloud-only", "Device-only", "Greedy-latency", "Round-robin", "Greedy-J (myopic)", "UQ-threshold", "Logged (random)", "ACO (proposed)"]
rows, Jw, Aseed0, traces0 = [], {}, {}, None
for name in METHODS:
    for s in SEEDS:
        met, J, A_full, tr = run_policy(name, s, trace=(name == "ACO (proposed)"))
        rows.append(dict(method=name, seed=s, **met)); Jw[(name, s)] = J
        if s == 0: Aseed0[name] = A_full
        if name == "ACO (proposed)": globals().setdefault("ALLTR", []).append(np.array(tr))
    print(name, "%.0fs" % (time.time() - T0))
per_seed = pd.DataFrame(rows); per_seed.to_csv(os.path.join(RES, "main_comparison_per_seed.csv"), index=False)
cols = ["accuracy_pct", "mean_latency_ms", "p95_latency_ms", "deadline_miss_pct", "energy_j", "exposure", "priv_violations",
        "J_per_frame", "ips_accuracy_pct", "offload_pct", "runtime_ms_per_window", "max_window_exposure"]
agg = per_seed.groupby("method", sort=False)[cols].agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]; agg = agg.reindex(METHODS)
agg.to_csv(os.path.join(RES, "table_main_comparison.csv"))

# significance: per-window J, ACO vs each baseline (seed-averaged per window), Wilcoxon signed-rank
sig = []
Jaco = np.mean([Jw[("ACO (proposed)", s)] for s in SEEDS], 0)
for name in METHODS[:-1]:
    Jb = np.mean([Jw[(name, s)] for s in SEEDS], 0)
    st, pv = wilcoxon(Jaco, Jb)
    sig.append(dict(baseline=name, mean_J_aco=Jaco.mean(), mean_J_base=Jb.mean(), rel_impr_pct=100 * (Jb.mean() - Jaco.mean()) / Jb.mean(),
                    windows_aco_better=int((Jaco < Jb).sum()), n_windows=len(Jaco), wilcoxon_p=pv))
pd.DataFrame(sig).to_csv(os.path.join(RES, "table_significance.csv"), index=False)

# ======================= 5. ablation =======================
print("ablation ...")
keep = windows; windows = keep[::3]
abl_cfg = [("ACO (full)", dict()), ("w/o uncertainty features", dict(planner="nouq")), ("w/o heuristic (beta=0)", dict(use_heur=False)),
           ("w/o local search", dict(use_ls=False)), ("w/o congestion model", dict(congestion=False)),
           ("arrival order (no UQ ordering)", dict(order_mode="arrival"))]
ab_rows = []
for nm, cfg in abl_cfg:
    for s in SEEDS[:3]:
        cfg2 = dict(cfg); pl = plan_noUQ if cfg2.pop("planner", None) == "nouq" else None
        met, J, _, _ = run_policy("ACO", s, planner=pl, **cfg2)
        ab_rows.append(dict(variant=nm, seed=s, **met))
windows = keep
ab = pd.DataFrame(ab_rows); ab.to_csv(os.path.join(RES, "ablation_per_seed.csv"), index=False)
abt = ab.groupby("variant", sort=False)[["accuracy_pct", "mean_latency_ms", "deadline_miss_pct", "J_per_frame"]].agg(["mean", "std"])
abt.columns = [f"{a}_{b}" for a, b in abt.columns]; abt.to_csv(os.path.join(RES, "table_ablation.csv"))

# privacy-rule study (README experiment 3): drop hard cloud ban + exposure budget
print("privacy study ...")
_fs = feasible_static; _eps = EPS_EXP
feasible_static = lambda i: np.ones(NA, bool); EPS_EXP = 1e9
priv_rows = []
keep = windows; windows = keep[::3]
for s in SEEDS[:3]:
    met, _, _, _ = run_policy("ACO", s); priv_rows.append(dict(variant="no privacy constraints", seed=s, **met))
feasible_static = _fs; EPS_EXP = _eps
for s in SEEDS[:3]:
    met, _, _, _ = run_policy("ACO", s); priv_rows.append(dict(variant="with privacy constraints (same windows)", seed=s, **met))
windows = keep
pv = pd.DataFrame(priv_rows); pv.to_csv(os.path.join(RES, "privacy_unconstrained_per_seed.csv"), index=False)

# ======================= 6. sensitivity: alpha x beta (subset), Pareto sweep over w_lat =======================
print("sensitivity ...")
keep = windows; windows = keep[::6]
sens = []
for a_ in [0.5, 1.0, 2.0]:
    for b_ in [0.0, 1.0, 2.0, 4.0]:
        for s in SEEDS[:2]:
            p = dict(ACO_P, alpha=a_, beta=b_)
            met, J, _, _ = run_policy("ACO", s, p=p)
            sens.append(dict(alpha=a_, beta=b_, seed=s, J=met["J_per_frame"]))
windows = keep
sens = pd.DataFrame(sens); sens.to_csv(os.path.join(RES, "sensitivity_alpha_beta_per_seed.csv"), index=False)
sens_t = sens.groupby(["alpha", "beta"]).J.mean().unstack(); sens_t.to_csv(os.path.join(RES, "table_sensitivity_alpha_beta.csv"))

pareto = []; keep = windows; windows = keep[::2]
for wl in [0.0, 0.03, 0.1, 0.3, 0.6, 1.2]:
    w2 = dict(W, lat=wl); register(planners[0], w2)
    for nm in ["ACO", "Greedy-J (myopic)"]:
        met, _, _, _ = run_policy(nm, 0); pareto.append(dict(policy="ACO (proposed)" if nm == "ACO" else nm, w_lat=wl, **met))
register(planners[0], W)
for tau_ in [0.15, 0.25, 0.35, 0.5, 0.7]:
    A_full = np.zeros(len(test), int)
    for idx in windows: A_full[idx] = uq_threshold(idx, TPL[id(planners[0])], tau_)
    met, _ = evaluate(A_full); pareto.append(dict(policy="UQ-threshold", w_lat=tau_, **met))
windows = keep
pareto = pd.DataFrame(pareto); pareto.to_csv(os.path.join(RES, "pareto_sweep.csv"), index=False)

# partition mix by entropy quartile
q = pd.qcut(test.device_pred_entropy.values, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
mix = pd.crosstab(q, pd.Categorical([ACTIONS[k] for k in Aseed0["ACO (proposed)"]], ACTIONS), normalize="index") * 100
mix.to_csv(os.path.join(RES, "aco_partition_mix_by_entropy_quartile.csv"))

# dataset summary table
summ = pd.DataFrame([
    dict(item="Frames (train 2025 / test 2026 H1)", value=f"{len(train)} / {len(test)}"),
    dict(item="Cameras / site types / tasks", value=f"{df.camera_id.nunique()} / 4 / 4"),
    dict(item="Error rate (all logged)", value=f"{100*df.prediction_error.mean():.1f}%"),
    dict(item="Error rate device_only / cloud_full", value=f"{100*df[df.partition_point=='device_only'].prediction_error.mean():.1f}% / {100*df[df.partition_point=='cloud_full'].prediction_error.mean():.1f}%"),
    dict(item="Median latency device_only / cloud_full (ms)", value=f"{df[df.partition_point=='device_only'].latency_ms.median():.0f} / {df[df.partition_point=='cloud_full'].latency_ms.median():.0f}"),
    dict(item="Frames with privacy_sensitivity >= 1", value=f"{100*(df.privacy_sensitivity>=1).mean():.1f}%"),
    dict(item="Missing values (lux / MC var / bandwidth)", value=f"{n_missing['illuminance_lux']} / {n_missing['device_mc_dropout_var']} / {n_missing['uplink_bandwidth_mbps']}"),
    dict(item="Outliers clipped (bandwidth / cloud RTT)", value=f"{n_out['uplink_bandwidth_mbps']} / {n_out['cloud_rtt_ms']}"),
])
summ.to_csv(os.path.join(RES, "table_dataset_summary.csv"), index=False)

# ======================= 7. figures =======================
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)

# Fig 2: ROC of error predictors
fig, ax = plt.subplots(figsize=(3.3, 2.5))
for sc, lab, col, ls in [(pm0, f"Full model (AUROC {model_q['auroc_full']:.3f})", ACC, "-"),
                         (pm_nouq, f"No UQ signals ({model_q['auroc_no_uq']:.3f})", GREY, "--"),
                         (pm_uq, f"UQ signals only ({model_q['auroc_uq_only']:.3f})", "#2c3e50", ":")]:
    f_, t_, _ = roc_curve(yt, sc); ax.plot(f_, t_, color=col, ls=ls, lw=1.3, label=lab)
f_, t_, _ = roc_curve(yt[dev_mask], ent[dev_mask]); ax.plot(f_, t_, color="#95a5a6", lw=1, ls="-.", label=f"Raw entropy, device frames ({model_q['auroc_entropy_raw_device_only']:.3f})")
ax.plot([0, 1], [0, 1], color="k", lw=0.5, alpha=0.4)
ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.legend(loc="lower right", frameon=False, fontsize=6.2)
ax.grid(alpha=0.25); save(fig, "fig2_error_roc")

# Fig 3: main comparison bars (horizontal, column width)
show = ["Cloud-only", "Device-only", "Greedy-latency", "Round-robin", "Greedy-J (myopic)", "UQ-threshold", "ACO (proposed)"]
lbl = ["Cloud-only", "Device-only", "Greedy-lat.", "Round-robin", "Greedy-J", "UQ-thresh.", "ACO (ours)"]
def fig3(agg):
    fig, axs = plt.subplots(1, 3, figsize=(3.4, 2.1), sharey=True)
    for ax, (c, t) in zip(axs, [("accuracy_pct", "Accuracy (%)"), ("mean_latency_ms", "Latency (ms)"), ("deadline_miss_pct", "Miss (%)")]):
        mu = agg.loc[show, c + "_mean"].values; sd = agg.loc[show, c + "_std"].values
        ax.barh(range(len(show)), mu, xerr=sd, color=[ACC if s.startswith("ACO") else GREY for s in show], height=0.7, capsize=1.5, error_kw=dict(lw=0.7))
        ax.set_xlabel(t, fontsize=7); ax.grid(axis="x", alpha=0.25); ax.tick_params(axis="x", labelsize=6.5)
        if c == "accuracy_pct": ax.set_xlim(40, 90)
    axs[0].set_yticks(range(len(show))); axs[0].set_yticklabels(lbl, fontsize=6.8); axs[0].invert_yaxis()
    fig.tight_layout(w_pad=0.4); save(fig, "fig3_method_comparison")
fig3(agg)

# Fig 4: convergence (cost normalised by myopic greedy-J per window)
TRS = np.array(ALLTR)                      # (seeds, windows, iters)
Jg = np.array([[solution_cost(GPL[id(planners[s])][idx], seq_policy(idx, TPL[id(planners[s])], (lambda G_, idx_: (lambda j: G_[idx_[j], :, 0]))(GPL[id(planners[s])], idx), np.argsort(-TPL[id(planners[s])]["P"][idx, 0]))[None], EDGE_UNITS)[0] for idx in windows] for s in SEEDS])
Jg_arr = np.array([[solution_cost(GPL[id(planners[s])][idx], seq_policy(idx, TPL[id(planners[s])], (lambda G_, idx_: (lambda j: G_[idx_[j], :, 0]))(GPL[id(planners[s])], idx)) [None], EDGE_UNITS)[0] for idx in windows] for s in SEEDS])
rel = TRS / Jg_arr[:, :, None]; curve = rel.mean(1); cm, cs = curve.mean(0), curve.std(0)
fig, ax = plt.subplots(figsize=(3.3, 2.2)); it = np.arange(1, TRS.shape[2] + 1)
ax.plot(it, cm, color=ACC, lw=1.4, label="ACO best-so-far (mean ± std, 5 seeds)"); ax.fill_between(it, cm - cs, cm + cs, color=ACC, alpha=0.25)
ax.axhline(1.0, color=GREY, ls="--", lw=1, label="Greedy-J, arrival order")
ax.axhline((Jg / Jg_arr).mean(), color="#2c3e50", ls=":", lw=1, label="Greedy-J, uncertainty order (warm start)")
ax.set_xlabel("ACO iteration"); ax.set_ylabel("Window cost / greedy-J cost"); ax.legend(frameon=False, fontsize=6.2); ax.grid(alpha=0.25)
save(fig, "fig4_aco_convergence")
conv = dict(final_rel_cost_mean=float(cm[-1]), final_rel_cost_std=float(cs[-1]), rel_cost_iter1=float(cm[0]),
            iter_to_99pct=int(np.argmax(cm <= cm[-1] + 0.01 * (cm[0] - cm[-1])) + 1), uq_order_greedy_rel=float((Jg / Jg_arr).mean()))

# Fig 6: Pareto accuracy vs latency
fig, ax = plt.subplots(figsize=(3.3, 2.5))
for pol, col, mk in [("ACO (proposed)", ACC, "o"), ("Greedy-J (myopic)", GREY, "s"), ("UQ-threshold", "#2c3e50", "^")]:
    d_ = pareto[pareto.policy == pol].sort_values("mean_latency_ms")
    ax.plot(d_.mean_latency_ms, d_.accuracy_pct, marker=mk, color=col, lw=1, ms=4, label=pol + (" (sweep $w_L$)" if pol != "UQ-threshold" else " (sweep $\\tau$)"))
for nm, mk in [("Cloud-only", "P"), ("Device-only", "X"), ("Greedy-latency", "D"), ("Round-robin", "v")]:
    r = per_seed[(per_seed.method == nm) & (per_seed.seed == 0)].iloc[0]
    ax.scatter(r.mean_latency_ms, r.accuracy_pct, marker=mk, color="k", s=22, label=nm, zorder=3)
ax.set_xlabel("Mean end-to-end latency (ms)"); ax.set_ylabel("Accuracy (%)"); ax.grid(alpha=0.25); ax.legend(fontsize=5.8, frameon=False, ncol=1, loc="lower right")
save(fig, "fig6_pareto")

# Fig 5: partition mix by entropy quartile, ACO vs greedy-J
mixg = pd.crosstab(q, pd.Categorical([ACTIONS[k] for k in Aseed0["Greedy-J (myopic)"]], ACTIONS), normalize="index") * 100
mixg.to_csv(os.path.join(RES, "greedyJ_partition_mix_by_entropy_quartile.csv"))
fig, ax = plt.subplots(figsize=(3.3, 2.3)); cols5 = ["#bdc3c7", "#f5b7b1", "#ec7063", ACC, "#641e16"]
x = np.arange(4); wdt = 0.38
for off, M, tag in [(-wdt / 2 - 0.01, mix, "A"), (wdt / 2 + 0.01, mixg, "G")]:
    bot = np.zeros(4)
    for k, an in enumerate(ACTIONS):
        v = M[an].values if an in M else np.zeros(4)
        ax.bar(x + off, v, wdt, bottom=bot, color=cols5[k], edgecolor="white", lw=0.4, label=SHORT[k] if tag == "A" else None); bot += v
for xi in x:
    ax.text(xi - wdt / 2 - 0.01, 101, "ACO", ha="center", fontsize=5.5); ax.text(xi + wdt / 2 + 0.01, 101, "Gr-J", ha="center", fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(mix.index.astype(str)); ax.set_ylim(0, 108)
ax.set_xlabel("Device predictive-entropy quartile"); ax.set_ylabel("Frames routed (%)")
ax.legend(ncol=5, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22)); save(fig, "fig5_partition_mix")

# Fig 7: alpha-beta sensitivity heatmap
fig, ax = plt.subplots(figsize=(3.0, 1.9))
im = ax.imshow(sens_t.values, cmap="Reds_r", aspect="auto")
ax.set_xticks(range(sens_t.shape[1])); ax.set_xticklabels([f"{c:g}" for c in sens_t.columns]); ax.set_yticks(range(sens_t.shape[0])); ax.set_yticklabels([f"{r:g}" for r in sens_t.index])
for i in range(sens_t.shape[0]):
    for j in range(sens_t.shape[1]):
        v = sens_t.values[i, j]; dark = (v - sens_t.values.min()) < 0.5 * (sens_t.values.max() - sens_t.values.min())
        ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=6.5, color="white" if dark else "black")
ax.set_xlabel(r"Heuristic exponent $\beta$"); ax.set_ylabel(r"Pheromone exponent $\alpha$"); plt.colorbar(im, ax=ax, label="Cost J per frame")
save(fig, "fig7_alpha_beta")

# Fig 1 flowchart (dot)
dot_src = os.path.join(HERE, "fig1_flowchart.dot")
subprocess.run(["dot", "-Tpng", "-Gdpi=300", dot_src, "-o", os.path.join(FIG, "fig1_flowchart.png")], check=True)
from PIL import Image
Image.open(os.path.join(FIG, "fig1_flowchart.png")).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"), quality=92)
shutil.copy(dot_src, os.path.join(RES, "fig1_flowchart.dot"))

# hyper-parameter table
hp = pd.DataFrame([dict(item="Window size B / edge capacity K_e / cloud quota Q_c", value=f"{B} frames / {K_EDGE:g} units / {Q_CLOUD}"),
                   dict(item="Edge units c_a (Split2, Split4, Edge)", value="0.5, 0.3, 1.0"),
                   dict(item="Exposure budget eps / congestion kappa", value=f"{EPS_EXP} / {KAPPA}"),
                   dict(item="Weights w_err, w_L, w_D, w_E", value=f"{W['err']}, {W['lat']}/100 ms, {W['miss']}, {W['en']}/J (x3 if battery<30%)"),
                   dict(item="Deadlines D (safety / other tasks)", value="250 ms / 500 ms"),
                   dict(item="ACO ants m, iterations T, stall", value=f"{ACO_P['ants']}, {ACO_P['iters']}, {ACO_P['stall']}"),
                   dict(item="ACO alpha, beta, rho, LS period", value=f"{ACO_P['alpha']}, {ACO_P['beta']}, {ACO_P['rho']}, {ACO_P['ls_every']}"),
                   dict(item="LightGBM planner (trees, lr, leaves)", value="300, 0.05, 15 (twin: 600, 0.03, 31)")])
hp.to_csv(os.path.join(RES, "table_hyperparameters.csv"), index=False)

metrics = dict(model_quality=model_q, convergence=conv, main=agg.reset_index().to_dict(orient="records"),
               significance=sig, ablation=abt.reset_index().to_dict(orient="records"),
               privacy_study=pv.groupby("variant")[cols].mean().to_dict(orient="index"), sensitivity_alpha_beta=sens_t.reset_index().to_dict(orient="records"),
               preprocessing=dict(missing={c: int(n_missing[c]) for c in miss_cols}, outliers_clipped=n_out, clip_bounds=clipq),
               n_windows=len(windows), n_test=len(test), n_train=len(train), runtime_s=time.time() - T0)
json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1, default=float)
print("done in %.0f s" % (time.time() - T0))
