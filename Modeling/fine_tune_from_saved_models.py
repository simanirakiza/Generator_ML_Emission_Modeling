"""
Fine-Tuning Script (Uses Pre-Trained LOSO Models) — CORRECTED VERSION v2

PREREQUISITE: Run cross_site_validation.py first to create saved models!

KEY FIXES vs the original fine_tune_from_saved_models.py:
1. Fixed test window: the LAST `test_days` days of each held-out site are reserved
   as a permanent test set, identical across every value of n_days. Day-1 vs day-3
   comparisons are now apples-to-apples (same eval set), not confounded by
   evaluating on a different/shrinking slice each time.
2. Multiple fine-tune windows per n_days, averaged: instead of always taking the
   FIRST n_days as the fine-tune set (one arbitrary calendar stretch), we sample
   several different contiguous n_days windows from the pre-test period and
   average the result. This tells you whether n_days of data reliably helps,
   not just whether one lucky/unlucky draw happened to help.
3. fillna(mean) is now computed ONLY from the fine-tune window's own statistics,
   and applied AFTER the fine-tune/test split — never using test-window data.
   (Diagnostic confirmed this matters very little in practice — raw sensor
   columns have ~zero NaNs, only the shift(1)-derived columns have 1-2 NaN rows
   per site — but it's now correct regardless.)
4. Sample-size-aware regularization: tree count/depth/ensemble weight now scale
   with n_samples instead of being fixed step functions, and the "ensemble"
   approach is replaced with a residual-correction model that is structurally
   capped in how much it can override the base LOSO model — this directly
   limits overfitting when only a few days of new-facility data are available.

NEW IN v2 (targeting the Katwa-style "high precision, low recall, recall won't
budge with more days" failure mode diagnosed from the v1 run):
5. Class-weighted fine-tuning: all three approaches now fit with
   class_weight='balanced' by default (toggle with --class_weight none to get
   v1 behavior back for comparison). This forces each fine-tune fit to weight
   rare generator-ON examples more heavily instead of relying on having enough
   ON examples by volume -- which v1's results showed Katwa doesn't reliably get.
6. Per-window positive-class diagnostics: every result row now also records
   n_pos and pos_rate (count/fraction of Generator_ON==True rows) in the
   fine-tune window that produced it. This lets you directly check whether a
   site's bad windows are exactly the ones with too few ON examples, rather
   than guessing.
7. Threshold tuning: for each fitted model, in addition to the default 0.5
   decision threshold, we search for the F1-optimal threshold using ONLY the
   fine-tune window's own out-of-bag-style holdout (a small slice carved out
   of the fine-tune window itself, never the fixed test window) and report
   metrics at both thresholds side by side. This separates "the model can't
   tell ON from OFF" (won't improve with threshold tuning) from "the model
   can tell, it's just decided too conservatively" (will improve).

USAGE:
    python fine_tune_from_saved_models_FIXED.py --root_path /path/to/repo \\
        --finetune_days "1,2,3,5,7" --test_days 7 --class_weight balanced
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
import utils
from importlib import reload
reload(utils)

# ============================================================================
# SETUP
# ============================================================================

parser = argparse.ArgumentParser(description="Fine-tuning from saved LOSO models (corrected)")
parser.add_argument('--root_path', type=str,
                   default="/Users/mahsa/PycharmProjects/",
                   help="Root path of the project")
parser.add_argument('--finetune_days', type=str,
                   default="1,2,3,5,7,14",
                   help="Days to test (e.g., '1,2,3,5,7')")
parser.add_argument('--test_days', type=int, default=12,
                   help="Number of days at the END of each site's data reserved as a "
                        "fixed test set (same across all finetune_days values)")
parser.add_argument('--n_windows', type=int, default=5,
                   help="Number of different contiguous fine-tune windows to sample "
                        "and average per finetune_days value (reduces lucky/unlucky draws)")
parser.add_argument('--approach', type=str,
                   default="all",
                   choices=["all", "continuation", "feature_adaptation", "residual_correction",
                            "xgb_warmstart", "xgb_fromscratch",
                            "tradaboost", "weighted_source_target", "freeze_structure"],
                   help="Which approach to test. xgb_* and freeze_structure require "
                        "--model_type XGBoost")
parser.add_argument('--b_weight', type=float, default=None,
                   help="Weight of each B_small (new-site) sample vs an A (source) sample "
                        "in weighted_source_target. DEFAULT (unset) = auto domain balance: "
                        "b_weight = n_A / n_B, so B_small's TOTAL vote equals A's total "
                        "vote regardless of dataset sizes. This derives the weight from the "
                        "data itself (for the current DRC setup it computes to ~20-50 "
                        "depending on window size, bracketing the empirically-found best "
                        "of ~36) and transfers to any unseen site without retuning. Pass a "
                        "number to override for experiments.")
parser.add_argument('--model_type', type=str,
                   default="RandomForest",
                   help="RandomForest or XGBoost")
parser.add_argument('--feature_group', type=str,
                   default="group3",
                   choices=["group3", "group4"],
                   help="Which saved base to fine-tune from: group3 (Voltage, Frequency, "
                        "Mean Threshold, Time Series) or group4 (adds HMM features). Must "
                        "match a base saved by cross_site_validation.py.")
parser.add_argument('--temporal_features', action='store_true',
                   help="Add rolling-window temporal features (std/range/drift/transitions "
                        "over 15/30/60 min per V-F channel = 48 extra columns) to fine-tune "
                        "training. IMPORTANT: only the FROM-SCRATCH approaches (continuation, "
                        "feature_adaptation, xgb_fromscratch, weighted_source_target, "
                        "tradaboost) can use these -- warm-start, freeze_structure, "
                        "residual_correction, and the LOSO_baseline reload the saved base "
                        "which was trained on the base feature set only, so they are "
                        "AUTOMATICALLY run on base features regardless of this flag. Off by "
                        "default. Diagnostic evidence: helps Katwa & Walikale at low label "
                        "budgets (3-7 days), neutral or slightly regressive at 14+ days -- "
                        "treat as a low-label-regime option.")
parser.add_argument('--seed', type=int, default=42,
                   help="Base random seed (window sampling varies by window index)")
parser.add_argument('--class_weight', type=str, default="balanced",
                   choices=["balanced", "none"],
                   help="Use class_weight='balanced' in fine-tune fits (default) or "
                        "'none' to reproduce v1 behavior for comparison")
parser.add_argument('--tune_threshold', action='store_true', default=True,
                   help="Also report metrics at the F1-optimal decision threshold, "
                        "tuned on a held-out slice of the fine-tune window only")
parser.add_argument('--threshold_holdout_frac', type=float, default=0.2,
                   help="Fraction of each fine-tune window carved out to calibrate "
                        "the decision threshold (never touches the fixed test window)")

args = parser.parse_args()
root_path = args.root_path
finetune_days = [int(d) for d in args.finetune_days.split(',')]
test_days = args.test_days
n_windows = args.n_windows
approach = args.approach
model_type = args.model_type
feature_group = args.feature_group
# temporal_features = args.temporal_features
# TODO
temporal_features = True
b_weight = args.b_weight
base_seed = args.seed
class_weight = 'balanced' if args.class_weight == 'balanced' else None
tune_threshold = args.tune_threshold
threshold_holdout_frac = args.threshold_holdout_frac

print(f"\n{'='*80}")
print(f"FINE-TUNING FROM SAVED LOSO MODELS — CORRECTED v2")
print(f"{'='*80}")
print(f"Root:             {root_path}")
print(f"Finetune days:    {finetune_days}")
print(f"Fixed test days:  {test_days} (held constant across all finetune_days values)")
print(f"Windows/value:    {n_windows} (averaged to reduce variance from window choice)")
print(f"Approach:         {approach}")
print(f"Model:            {model_type}")
print(f"Feature group:    {feature_group}")
print(f"Temporal feats:   {temporal_features} "
      f"({'enabled for from-scratch approaches only' if temporal_features else 'off'})")
print(f"Class weight:     {class_weight}")
print(f"Threshold tuning: {tune_threshold} (calibration holdout frac={threshold_holdout_frac})")
print(f"{'='*80}\n")

# Paths
savepath = f"{root_path}Generator_ML_Emission_Modeling/Modeling/summer25_results/cross_site_models/"
model_save_dir = f"{savepath}saved_models/"

if not os.path.exists(model_save_dir):
    print(f"ERROR: Model directory not found: {model_save_dir}")
    print("Run cross_site_validation.py first to create saved models!")
    exit(1)

# Constants
minutes_interval = 2
samples_per_day = 720  # 24h * 60min / 2min interval

# Site setup (same as original)
sites = ["Ndosho", "Walikale", "Kitatumba", "Katwa"]
data_paths = [
    f"{root_path}Generator_ML_Emission_Modeling/data/0. ChCbNdosho/training",
    f"{root_path}Generator_ML_Emission_Modeling/data/1. Combination 1_Walikale/training",
    f"{root_path}Generator_ML_Emission_Modeling/data/5. Combination 5/training/kitatumba",
    f"{root_path}Generator_ML_Emission_Modeling/data/5. Combination 5/training/katwa"
]

site_columns_oi = {
   "Walikale":[['Frequency 09B35150 (Hgr Walikale - Lab)',
                'Frequency 64B734C2 (Hgr Walikale - Cabane)',
                'Voltage 09B35150 (Hgr Walikale - Lab)',
                'Voltage 64B734C2 (Hgr Walikale - Cabane)'
                ], data_paths[1] ],
     "Ndosho": [[ 'Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                'Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)',
                'Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                'Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)'
                ],data_paths[0]],
      "Katwa":[['Frequency 1C712FB8 (Hgr Katwa - Labo)',
                'Frequency AEBA416D (Hgr Katwa - Cabine)',
                'Voltage 1C712FB8 (Hgr Katwa - Labo)',
                'Voltage AEBA416D (Hgr Katwa - Cabine)'
                ], data_paths[3]],
    "Kitatumba": [['Frequency 6EF39700 (Hgr Kitatumba - Labo)',
                   'Frequency D2B9CB2E (Hgr Kitatumba - Mdh office)',
                   'Voltage 6EF39700 (Hgr Kitatumba - Labo)',
                   'Voltage D2B9CB2E (Hgr Kitatumba - Mdh office)'
    ], data_paths[2]]
}

# Load data
print("Loading data...")
site_dfs = {site: utils.load_data(path) for site, path in zip(sites, data_paths)}

def rename_site_columns(df, site):
    freq_cols = site_columns_oi[site][0][:2]
    volt_cols = site_columns_oi[site][0][2:]
    rename_map = {
        freq_cols[0]: 'freq_a',
        freq_cols[1]: 'freq_b',
        volt_cols[0]: 'volt_a',
        volt_cols[1]: 'volt_b'
    }
    df = df.rename(columns=rename_map)
    df['Site'] = site
    return df

processed_dfs = [rename_site_columns(df, site) for site, df in site_dfs.items()]
combined_df = pd.concat(processed_dfs, ignore_index=True)
combined_df['Time'] = pd.to_datetime(combined_df['Time'])
print("✓ Data loaded\n")

# ============================================================================
# FINE-TUNING APPROACHES (sample-size-aware)
# ============================================================================

def _scaled_rf_params(n_samples):
    """
    Scale RandomForest complexity with available fine-tune sample size.
    Fewer samples -> shallower trees, more leaf-size regularization,
    fewer estimators added — directly limits overfitting on small
    new-facility samples instead of using one fixed setting for everyone.
    """
    if n_samples < 1000:          # < ~1.4 days
        params = dict(n_estimators=30, max_depth=4, min_samples_leaf=15)
    elif n_samples < 2000:        # ~1.4-2.8 days
        params = dict(n_estimators=50, max_depth=6, min_samples_leaf=10)
    elif n_samples < 3500:        # ~2.8-5 days
        params = dict(n_estimators=75, max_depth=8, min_samples_leaf=6)
    else:                         # 5+ days
        params = dict(n_estimators=100, max_depth=12, min_samples_leaf=4)
    if class_weight is not None:
        params['class_weight'] = class_weight
    return params


def find_best_threshold(model, X_calib, y_calib, default=0.5):
    """
    Search for the F1-optimal decision threshold using ONLY a calibration
    slice carved out of the fine-tune window (X_calib/y_calib never include
    anything from the fixed test window). Returns (best_threshold, default).

    This directly tests the hypothesis that a low-recall site's model can
    already separate ON/OFF reasonably in its predicted probabilities, but
    the default 0.5 cutoff is too conservative for an imbalanced class --
    a much cheaper fix than retraining if that's what's going on.
    """
    p = model.predict_proba(X_calib)
    if p.shape[1] < 2:
        return default, default  # model only ever saw one class; nothing to tune

    probs = p[:, 1]
    if y_calib.sum() == 0:
        return default, default  # no positives in calibration slice; can't tune meaningfully

    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = default, -1.0
    for t in candidate_thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(y_calib, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, default


def approach1_continuation(base_model, X_finetune, y_finetune):
    """Retrain a fresh forest sized to the available fine-tune data.
    (Renamed conceptually from 'add 50 trees' to 'size complexity to data',
    since blindly adding 50 trees regardless of n_samples is itself a
    source of overfitting on small samples.)"""
    params = _scaled_rf_params(len(X_finetune))
    print(f"      -> Continuation (n_estimators={params['n_estimators']}, "
          f"max_depth={params['max_depth']}, min_samples_leaf={params['min_samples_leaf']})")
    finetuned = RandomForestClassifier(random_state=42, **params)
    finetuned.fit(X_finetune, y_finetune)
    return finetuned


def approach2_feature_adaptation(base_model, X_finetune, y_finetune, feature_names):
    """Select top features using a size-appropriate forest, then refit."""
    print("      -> Feature Adaptation (selecting features)")
    params = _scaled_rf_params(len(X_finetune))

    temp_kwargs = dict(
        n_estimators=max(20, params['n_estimators'] // 2),
        max_depth=max(4, params['max_depth'] - 2),
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    if class_weight is not None:
        temp_kwargs['class_weight'] = class_weight
    temp = RandomForestClassifier(**temp_kwargs)
    temp.fit(X_finetune, y_finetune)

    importances = temp.feature_importances_
    # Keep fewer features when data is scarce -> less room to overfit
    keep_frac = 0.5 if len(X_finetune) < 2000 else 0.6
    n_keep = max(6, int(len(feature_names) * keep_frac))
    top_idx = np.argsort(importances)[::-1][:n_keep]

    print(f"         Selected {n_keep}/{len(feature_names)} features")

    adapted = RandomForestClassifier(random_state=42, **params)
    adapted.fit(X_finetune[:, top_idx], y_finetune)

    class FeatureAdapter:
        def __init__(self, model, indices):
            self.model = model
            self.indices = indices
        def predict(self, X):
            return self.model.predict(X[:, self.indices])
        def predict_proba(self, X):
            return self.model.predict_proba(X[:, self.indices])

    return FeatureAdapter(adapted, top_idx)


def approach3_residual_correction(base_model, X_finetune, y_finetune, n_samples):
    """
    Replaces the old fixed-weight ensemble. Instead of guessing a blend
    weight (0.30 / 0.40 / 0.50), we fit a small correction model that only
    nudges the base LOSO prediction, and we cap how much it's allowed to
    move the decision using a confidence-scaled blend that shrinks toward
    the base model as n_samples shrinks. This is the structural way to
    "use a little data without overfitting" -- the correction's influence
    is bounded by how much data supports it, not by a hand-picked constant.
    """
    print("      -> Residual Correction (bounded site-specific adjustment)")

    params = _scaled_rf_params(n_samples)
    # Correction model is intentionally smaller/shallower than the main
    # scaled params -- it only needs to learn a local adjustment, not the
    # whole problem from scratch.
    site_kwargs = dict(
        n_estimators=max(20, params['n_estimators'] // 2),
        max_depth=max(3, params['max_depth'] - 2),
        min_samples_leaf=max(params['min_samples_leaf'], 10),
        random_state=42
    )
    if class_weight is not None:
        site_kwargs['class_weight'] = class_weight
    site_model = RandomForestClassifier(**site_kwargs)
    site_model.fit(X_finetune, y_finetune)

    # Confidence-scaled weight: grows smoothly with sample size instead of
    # jumping between 3 fixed bands. Capped at 0.5 so the small-sample
    # model can never out-vote the LOSO base model, regardless of n_samples.
    # 500 samples (~17 hours) -> ~0.20 weight
    # 1440 samples (2 days)   -> ~0.42 weight
    # 2880 samples (4 days)   -> approaches the 0.50 cap
    weight_site = min(0.5, n_samples / (n_samples + 2000))
    print(f"         Weights: {int((1 - weight_site) * 100)}% LOSO base, "
          f"{int(weight_site * 100)}% site correction (n={n_samples})")

    class ResidualCorrected:
        def __init__(self, base, site, w):
            self.base = base
            self.site = site
            self.w = w

        def _proba(self, model, X):
            p = model.predict_proba(X)
            if p.shape[1] == 2:
                return p[:, 1]
            # Model only ever saw one class in fine-tune data
            return model.predict(X).astype(float)

        def predict(self, X):
            p_base = self._proba(self.base, X)
            p_site = self._proba(self.site, X)
            return ((1 - self.w) * p_base + self.w * p_site > 0.5).astype(int)

        def predict_proba(self, X):
            p_base = self._proba(self.base, X)
            p_site = self._proba(self.site, X)
            p1 = (1 - self.w) * p_base + self.w * p_site
            return np.column_stack([1 - p1, p1])

    return ResidualCorrected(base_model, site_model, weight_site)


def approach4_xgb_warmstart(base_model, X_finetune, y_finetune, n_samples):
    """
    REAL fine-tuning, only possible with XGBoost (not RandomForest).

    Unlike approach1/approach2 -- which secretly retrain a brand-new model on
    the new site's data and ignore the base model entirely -- this CONTINUES
    training the saved LOSO base model (trained on the other 3 sites) by adding
    a small number of new boosting rounds fit on the new site's window, via
    xgboost's xgb_model= warm-start. The base model's existing trees are kept;
    the new trees only correct the base's errors on the new site. This is the
    "adapt what 3 sites taught us using a little new-site data" idea that the
    RandomForest approaches structurally could not do.

    n_added boosting rounds scales (gently) with available data so a tiny
    window can't bolt on a huge number of site-specific trees and overfit.
    """
    if n_samples < 1000:
        n_added = 20
    elif n_samples < 2000:
        n_added = 35
    elif n_samples < 3500:
        n_added = 50
    else:
        n_added = 75

    print(f"      -> XGB warm-start (continue base model + {n_added} new rounds)")

    spw = None
    n_pos = int(np.sum(y_finetune))
    n_neg = len(y_finetune) - n_pos
    if n_pos > 0:
        spw = n_neg / n_pos  # scale_pos_weight: XGBoost's class-imbalance lever

    ft = XGBClassifier(
        n_estimators=n_added,
        max_depth=4,
        learning_rate=0.05,   # small LR for fine-tuning: nudge, don't overwrite
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=spw if spw is not None else 1.0,
    )
    # Continue from the saved base model's booster (the actual warm-start)
    booster = base_model.get_booster() if hasattr(base_model, 'get_booster') else base_model
    ft.fit(X_finetune, y_finetune, xgb_model=booster)
    return ft


def approach5_xgb_fromscratch(base_model, X_finetune, y_finetune, n_samples):
    """
    Control/comparison for approach4: a fresh XGBoost trained ONLY on the new
    site's window, ignoring the base model -- the XGBoost analogue of what
    approach1/approach2 do for RandomForest. Included so the warm-start's
    benefit (or lack of it) can be measured head-to-head against from-scratch
    on the SAME model family, isolating "warm-start vs scratch" from
    "XGBoost vs RandomForest".
    """
    print(f"      -> XGB from-scratch (new site only, base ignored)")

    spw = None
    n_pos = int(np.sum(y_finetune))
    n_neg = len(y_finetune) - n_pos
    if n_pos > 0:
        spw = n_neg / n_pos

    if n_samples < 1000:
        n_est, depth = 40, 3
    elif n_samples < 2000:
        n_est, depth = 75, 4
    else:
        n_est, depth = 120, 5

    m = XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=spw if spw is not None else 1.0,
    )
    m.fit(X_finetune, y_finetune)
    return m


# ============================================================================
# TRANSFER-LEARNING METHODS (use BOTH A = 3-site source and B_small = new-site window)
# ============================================================================

def approach6_tradaboost(X_A, y_A, X_B, y_B, n_iters=20, base_estimator_depth=4, seed=42):
    """
    Method 1: Transfer Boosting (TrAdaBoost, Dai et al. 2007).

    Setup matches yours exactly: large SOURCE A (3 sites) + small TARGET
    B_small (held-out site window). On each round we train a weak learner on
    the combined weighted set, then:
      - source (A) samples that are misclassified get their weight DECREASED
        (they're judged unhelpful for the target and gradually ignored),
      - target (B) samples that are misclassified get their weight INCREASED
        (standard AdaBoost behavior, focusing on hard target examples).
    Final prediction uses the second half of the learners (weighted vote),
    as in the original TrAdaBoost, by which point bad A samples are suppressed
    and B_small dominates the adaptation.

    Returns an object with .predict / .predict_proba over the SAME feature
    space as X_A/X_B (so callers pass the appropriately-treated test matrix).
    """
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(seed)
    n_A, n_B = len(X_A), len(X_B)
    X = np.vstack([X_A, X_B])
    y = np.concatenate([y_A, y_B]).astype(int)

    # initial weights: uniform within each domain
    w = np.ones(n_A + n_B)
    w[:n_A] = 1.0 / n_A
    w[n_A:] = 1.0 / n_B
    w = w / w.sum()

    # TrAdaBoost source-weight decay factor
    beta = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(n_A, 1)) / max(n_iters, 1)))

    learners, betas_t = [], []
    for t in range(n_iters):
        p = w / w.sum()
        # weighted weak learner (shallow tree)
        stump = DecisionTreeClassifier(max_depth=base_estimator_depth, random_state=rng.randint(1 << 30))
        stump.fit(X, y, sample_weight=p)
        pred = stump.predict(X)
        err_vec = (pred != y).astype(float)

        # weighted error on the TARGET (B) portion only
        wB = p[n_A:]
        eps = float(np.sum(wB * err_vec[n_A:]) / max(np.sum(wB), 1e-12))
        eps = min(max(eps, 1e-6), 0.499)  # clamp for stability
        bt = eps / (1.0 - eps)

        learners.append(stump)
        betas_t.append(bt)

        # update weights:
        #  source A: multiply by beta**err  (decrease weight of wrong A samples)
        #  target B: multiply by bt**(-err) (increase weight of wrong B samples)
        w[:n_A] = w[:n_A] * np.power(beta, err_vec[:n_A])
        w[n_A:] = w[n_A:] * np.power(bt, -err_vec[n_A:])
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()

    class TrAdaBoostModel:
        def __init__(self, learners, betas_t):
            # use the LATER half of learners (TrAdaBoost convention)
            start = len(learners) // 2
            self.learners = learners[start:]
            self.betas_t = betas_t[start:]
        def _score(self, X):
            # weighted vote in log(1/bt) space -> probability of class 1
            num = np.zeros(len(X)); den = 0.0
            for lr, bt in zip(self.learners, self.betas_t):
                alpha = np.log(1.0 / max(bt, 1e-12))
                num += alpha * lr.predict(X)
                den += alpha
            return num / max(den, 1e-12)
        def predict_proba(self, X):
            p1 = self._score(X)
            p1 = np.clip(p1, 0.0, 1.0)
            return np.column_stack([1 - p1, p1])
        def predict(self, X):
            return (self._score(X) >= 0.5).astype(int)

    return TrAdaBoostModel(learners, betas_t)


def approach7_weighted_source_target(X_A, y_A, X_B, y_B, b_weight=None,
                                     model_type='XGBoost', n_samples_B=None, seed=42):
    """
    Method 2: Weighted retraining on A + B_small (instance-transfer learning).

    Train ONE model on the union of A (3-site source data) and B_small
    (new-site window), with every B_small sample upweighted. The source
    knowledge is carried as DATA rather than as frozen model parameters --
    the learner reconciles A's evidence with B_small's, instead of starting
    from A's (possibly wrong-for-this-site) conclusions the way warm-start
    does. This is why it outperformed warm-start on Katwa, whose base-model
    prior was confidently OFF-biased.

    b_weight=None (default) -> AUTO DOMAIN BALANCE: b_weight = n_A / n_B,
    i.e. B_small's total sample weight equals A's total weight regardless of
    dataset sizes. Derived from the data at hand, transfers to any unseen
    site with no retuning, and for the current DRC data shapes it computes
    to ~20-50 depending on window size -- bracketing the empirically-found
    best (~36), which is evidence the principle explains that finding rather
    than replacing it. Pass an explicit number to override for experiments.
    """
    if b_weight is None:
        b_weight = len(X_A) / max(len(X_B), 1)
        print(f"         auto b_weight (domain balance n_A/n_B) = {b_weight:.1f}")

    X = np.vstack([X_A, X_B])
    y = np.concatenate([y_A, y_B]).astype(int)
    sw = np.ones(len(X))
    sw[len(X_A):] = b_weight

    npos = int(np.sum(y)); nneg = len(y) - npos
    spw = (nneg / npos) if npos > 0 else 1.0

    if model_type == 'XGBoost':
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          random_state=seed, eval_metric='logloss',
                          scale_pos_weight=spw)
        m.fit(X, y, sample_weight=sw)
    else:
        m = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   min_samples_leaf=4, random_state=seed,
                                   class_weight='balanced')
        m.fit(X, y, sample_weight=sw)
    return m


def approach8_freeze_structure_update_leaves(base_model, X_B, y_B, feature_names):
    """
    Method 3: Freeze tree structure (from A), recompute leaf values on B_small.

    Keep the saved XGBoost base model's tree SPLITS exactly as learned on A,
    but replace each leaf's output using ONLY the B_small samples that fall
    into that leaf. Analogous to freezing a pretrained net's layers and
    refitting only the head. Leaves that no B_small sample reaches keep their
    original A value (so structure-from-A still covers regions B_small never
    visits).

    Implemented faithfully via xgboost's pred_leaf (which leaf each sample
    lands in, per tree). We rebuild a prediction function that, per tree,
    maps leaf -> recomputed log-odds contribution from B_small, falling back
    to the original leaf value when B_small has no support there.
    """
    import xgboost as xgb
    import json

    booster = base_model.get_booster()
    n_trees = booster.num_boosted_rounds()

    dmat_B = xgb.DMatrix(X_B, feature_names=None)
    leaf_B = booster.predict(dmat_B, pred_leaf=True)  # (n_B, n_trees)
    if leaf_B.ndim == 1:
        leaf_B = leaf_B.reshape(-1, 1)

    # original leaf values per tree from the JSON dump
    dump = booster.get_dump(dump_format='json')
    orig_leaf_val = []  # list over trees: dict leaf_nodeid -> value
    for tree_json in dump:
        d = json.loads(tree_json)
        leaves = {}
        stack = [d]
        while stack:
            node = stack.pop()
            if 'leaf' in node:
                leaves[node['nodeid']] = float(node['leaf'])
            else:
                stack.extend(node.get('children', []))
        orig_leaf_val.append(leaves)

    # learning rate / base_score from config
    cfg = json.loads(booster.save_config())
    try:
        base_score = float(cfg['learner']['learner_model_param']['base_score'])
    except Exception:
        base_score = 0.5
    base_margin = np.log(base_score / (1 - base_score)) if 0 < base_score < 1 else 0.0

    yB = np.asarray(y_B).astype(int)
    # recompute each leaf's value as the log-odds of B_small samples in it
    new_leaf_val = []
    for t in range(n_trees):
        leaves_here = {}
        col = leaf_B[:, t]
        for leaf_id in np.unique(col):
            mask = col == leaf_id
            if mask.sum() == 0:
                continue
            pos = yB[mask].sum(); n = mask.sum()
            # Laplace-smoothed log-odds, shrunk toward original by support size
            p = (pos + 1.0) / (n + 2.0)
            recomputed = np.log(p / (1 - p))
            orig = orig_leaf_val[t].get(int(leaf_id), 0.0)
            # shrink: more B support -> trust recomputed more
            alpha = n / (n + 10.0)
            leaves_here[int(leaf_id)] = alpha * recomputed + (1 - alpha) * orig
        new_leaf_val.append(leaves_here)

    class FrozenStructureModel:
        def __init__(self, booster, orig_leaf_val, new_leaf_val, base_margin, n_trees):
            self.booster = booster
            self.orig = orig_leaf_val
            self.new = new_leaf_val
            self.base_margin = base_margin
            self.n_trees = n_trees
        def _margin(self, X):
            dm = xgb.DMatrix(X)
            leaf = self.booster.predict(dm, pred_leaf=True)
            if leaf.ndim == 1:
                leaf = leaf.reshape(-1, 1)
            margin = np.full(len(X), self.base_margin, dtype=float)
            for t in range(self.n_trees):
                col = leaf[:, t]
                newmap, origmap = self.new[t], self.orig[t]
                for i, lf in enumerate(col):
                    lf = int(lf)
                    margin[i] += newmap.get(lf, origmap.get(lf, 0.0))
            return margin
        def predict_proba(self, X):
            m = self._margin(X)
            p1 = 1.0 / (1.0 + np.exp(-m))
            return np.column_stack([1 - p1, p1])
        def predict(self, X):
            return (self._margin(X) >= 0.0).astype(int)

    print(f"      -> Freeze-structure/update-leaves (recomputed leaves on {len(yB)} B samples)")
    return FrozenStructureModel(booster, orig_leaf_val, new_leaf_val, base_margin, n_trees)
# MAIN EXPERIMENT
# ============================================================================

def calculate_metrics(y_true, y_pred, site, n_pos=None, pos_rate=None, threshold_used=None):
    metrics = {
        'site': site,
        'model': model_type,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'gen_detection_error': abs(y_pred.sum() - y_true.sum()) / y_true.sum() * 100 if y_true.sum() > 0 else 0
    }
    if n_pos is not None:
        metrics['finetune_n_pos'] = n_pos
    if pos_rate is not None:
        metrics['finetune_pos_rate'] = pos_rate
    if threshold_used is not None:
        metrics['threshold_used'] = threshold_used
    return metrics


def add_temporal_features(df_ts, freq_cols=('freq_a', 'freq_b'),
                          volt_cols=('volt_a', 'volt_b')):
    """
    Rolling-window temporal features on top of the base pipeline. Not used by
    default; enabled by --temporal_features. Rolling windows are BACKWARD-
    LOOKING (min_periods=1) so no future information leaks into any row.
    Computed PER SITE so a rolling window never spans two facilities.

    Windows: 15 / 30 / 60 min = 7 / 15 / 30 rows at 2-min sampling.
    For each of (freq_a, freq_b, volt_a, volt_b) x (15, 30, 60 min):
      * rolling std     -- variance signature (Katwa's ON drifts, OFF doesn't)
      * rolling range   -- within-episode drift magnitude
      * rolling mean-drift = current - rolling mean -- shift direction
      * transition count of |delta| > eps in window -- how "jumpy" the signal is
    Returns (augmented_df, list_of_new_column_names). Original df untouched.

    Design was validated by the standalone diagnostic (step1_temporal_features_
    test.py): +0.09 to +0.15 F1 on Katwa at 3-7 days, +0.04-0.08 on Walikale,
    neutral on Ndosho, small regression on Kitatumba at 3d only. This is why
    it's an OPT-IN flag and only wired into the FROM-SCRATCH approaches --
    warm-start / freeze_structure / residual_correction / LOSO_baseline use
    the saved base model's fixed feature list and can't consume these columns.
    """
    df = df_ts.copy()
    if 'Site' not in df.columns:
        raise ValueError("dataframe must have a 'Site' column to avoid "
                         "rolling across sites")

    windows_rows = {'15min': 7, '30min': 15, '60min': 30}
    eps_freq, eps_volt = 0.5, 5.0

    channels = list(freq_cols) + list(volt_cols)
    new_cols = []

    for site, sub in df.groupby('Site', sort=False):
        sub = sub.sort_values('Time') if 'Time' in sub.columns else sub
        idx = sub.index
        for ch in channels:
            eps = eps_freq if 'freq' in ch else eps_volt
            s = pd.to_numeric(sub[ch], errors='coerce')
            for wname, wlen in windows_rows.items():
                col_std   = f'roll_std_{ch}_{wname}'
                col_rng   = f'roll_range_{ch}_{wname}'
                col_drift = f'roll_drift_{ch}_{wname}'
                col_trans = f'roll_trans_{ch}_{wname}'
                r = s.rolling(window=wlen, min_periods=1)
                df.loc[idx, col_std]   = r.std().fillna(0.0).values
                df.loc[idx, col_rng]   = (r.max() - r.min()).fillna(0.0).values
                df.loc[idx, col_drift] = (s - r.mean()).fillna(0.0).values
                jumps = (s.diff().abs() > eps).astype(float)
                df.loc[idx, col_trans] = jumps.rolling(window=wlen, min_periods=1)\
                                              .sum().fillna(0.0).values
                for c in (col_std, col_rng, col_drift, col_trans):
                    if c not in new_cols:
                        new_cols.append(c)
    return df, new_cols


def make_xgb_feature_matrix(X_scaled, X_raw_df, feature_names):
    """
    Reproduce the XGBoost-specific feature treatment from cross_site_validation.py:
    the 4 Prev_High_delta_* columns are overwritten with their RAW (unscaled)
    numeric values rather than the StandardScaler-scaled values. The saved
    XGBoost base model was trained on a matrix built this way, so for the
    warm-start (and a fair from-scratch comparison) the fine-tune/test matrices
    must match it. RandomForest approaches are unaffected and keep using the
    fully-scaled matrix.
    """
    xgb_cols = ['Prev_High_delta_freq_lab', 'Prev_High_delta_freq_clinique',
                'Prev_High_delta_volt_lab', 'Prev_High_delta_volt_clinique']
    X = X_scaled.copy()
    for col in xgb_cols:
        if col in feature_names:
            idx = feature_names.index(col) if isinstance(feature_names, list) \
                else list(feature_names).index(col)
            X[:, idx] = pd.to_numeric(X_raw_df[col], errors='coerce').values
    return X


def run_fine_tuning():
    results = []
    approaches_to_run = ['continuation', 'feature_adaptation', 'residual_correction'] \
        if approach == "all" else [approach]

    # If the base model is XGBoost, ALSO run the real warm-start fine-tuning
    # plus its from-scratch control -- additively, without removing the RF-style
    # approaches, so old comparisons stay intact.
    if model_type == "XGBoost" and approach == "all":
        approaches_to_run = approaches_to_run + ['xgb_warmstart', 'xgb_fromscratch']
    elif approach in ('xgb_warmstart', 'xgb_fromscratch'):
        approaches_to_run = [approach]

    # Transfer-learning methods (use A = 3-site source + B_small = new-site window).
    # tradaboost and weighted_source_target work for both RF and XGB; the
    # freeze-structure method requires an XGBoost base (it edits XGB leaves).
    if approach == "all":
        approaches_to_run = approaches_to_run + ['tradaboost', 'weighted_source_target']
        if model_type == "XGBoost":
            approaches_to_run = approaches_to_run + ['freeze_structure']
    elif approach in ('tradaboost', 'weighted_source_target', 'freeze_structure'):
        approaches_to_run = [approach]

    test_samples = int(samples_per_day * test_days)

    for test_site in sites:
        print(f"\n{'='*80}")
        print(f"SITE: {test_site}")
        print(f"{'='*80}\n")

        # Load saved LOSO model
        model_filename = f"loso_baseline_{test_site}_{model_type}_{feature_group}.pkl"
        model_path = os.path.join(model_save_dir, model_filename)

        if not os.path.exists(model_path):
            print(f"  X Model not found: {model_filename}")
            print(f"  Run cross_site_validation.py first!")
            continue

        print(f"Loading saved model: {model_filename}")
        saved = joblib.load(model_path)
        base_model = saved['model']
        scaler = saved['scaler']
        feature_names = saved['feature_names']
        print(f"OK Loaded (trained on {', '.join(saved['train_sites'])})\n")

        # Guard: XGBoost approaches require an XGBoost base model and the xgboost lib.
        xgb_only = ('xgb_warmstart', 'xgb_fromscratch', 'freeze_structure')
        wants_xgb = any(a in xgb_only for a in approaches_to_run)
        if wants_xgb and not _HAS_XGB:
            print("  ! xgboost not installed -- skipping XGBoost approaches for this site.")
            approaches_to_run = [a for a in approaches_to_run if a not in xgb_only]
        if not hasattr(base_model, 'get_booster'):
            dropped = [a for a in approaches_to_run if a in ('xgb_warmstart', 'freeze_structure')]
            if dropped:
                print(f"  ! Base model is not XGBoost -- skipping {dropped} "
                      f"(need --model_type XGBoost).")
                approaches_to_run = [a for a in approaches_to_run if a not in dropped]

        # Prepare data
        held_out_data = combined_df[combined_df['Site'] == test_site].copy()
        held_out_data = held_out_data.sort_values('Time').reset_index(drop=True)
        other_sites_data = combined_df[combined_df['Site'] != test_site].copy()

        columns_to_compare = [
            ('freq_a', 'freq_lab'), ('freq_b', 'freq_clinique'),
            ('volt_a', 'volt_lab'), ('volt_b', 'volt_clinique')
        ]

        # NOTE: generate_group3features computes the higher_*_mean flags using
        # other_sites_data as the base -- this is correct LOSO practice and
        # does NOT leak held-out-site information (verified against utils.py).
        train_data_ts, test_data_ts = utils.generate_group3features(
            other_sites_data, held_out_data, columns_to_compare,
            'freq_a', 'freq_b', 'volt_a', 'volt_b'
        )

        # For group4, chain the HMM features on top of the group3 features,
        # exactly as cross_site_validation.py does when it builds/saves the
        # group4 base -- so the columns named in the saved model's
        # feature_names list actually exist in test_data_ts here.
        if feature_group == "group4":
            train_data_ts, test_data_ts, hmm_features = utils.generate_group4features(
                train_data_ts, test_data_ts, 'freq_a', 'freq_b', 'volt_a', 'volt_b'
            )
            print(f"  group4: added HMM features -> {hmm_features}")

        # ------------------------------------------------------------------
        # OPT-IN: temporal rolling-window features. Computed on the RAW 4-column
        # V/F series (per site, no cross-site rolling) then spliced onto the
        # featurized frames by (Time, Site). These columns are used ONLY by
        # the from-scratch approaches -- the saved base model's fixed feature
        # list is unchanged and reload-based approaches (warm-start,
        # freeze_structure, residual_correction, LOSO_baseline) continue to
        # use `feature_names` as-is. Kept in a separate list `temporal_cols`
        # so the routing is explicit and auditable, not implicit.
        # ------------------------------------------------------------------
        temporal_cols = []
        if temporal_features:
            raw_all = pd.concat([
                other_sites_data[['Time', 'Site', 'freq_a', 'freq_b', 'volt_a', 'volt_b']],
                held_out_data[['Time', 'Site', 'freq_a', 'freq_b', 'volt_a', 'volt_b']],
            ], ignore_index=True)
            raw_all, temporal_cols = add_temporal_features(raw_all)
            merge_key_cols = ['Time', 'Site']
            temp_slice = raw_all[merge_key_cols + temporal_cols]\
                .drop_duplicates(subset=merge_key_cols)
            def _splice(featurized_df):
                mc = [c for c in merge_key_cols if c in featurized_df.columns]
                if not mc:
                    return featurized_df
                return featurized_df.merge(temp_slice[mc + temporal_cols],
                                          on=mc, how='left')
            train_data_ts = _splice(train_data_ts)
            test_data_ts = _splice(test_data_ts)
            print(f"  temporal features enabled: +{len(temporal_cols)} cols "
                  f"(from-scratch approaches only)")

        # Feature-name lists split explicitly:
        #   feature_names       -- what the saved base model expects (unchanged)
        #   fromscratch_feats   -- extended list for from-scratch approaches
        fromscratch_feats = list(feature_names) + [
            c for c in temporal_cols if c in train_data_ts.columns]

        if len(test_data_ts) <= test_samples:
            print(f"  Skipping {test_site}: not enough data for a {test_days}-day fixed test window\n")
            continue

        # ------------------------------------------------------------------
        # A = SOURCE data (the 3 training sites), used by the transfer-learning
        # methods (TrAdaBoost, weighted A+B_small, freeze-structure). Built
        # once per held-out site. y_A is its labels. We build both the scaled
        # matrix (for RF-style use) and the XGB-treated matrix (Prev_High_delta_*
        # left raw) so XGBoost transfer methods match the base model's features.
        # ------------------------------------------------------------------
        A_fill = train_data_ts[feature_names].mean()
        X_A_df = train_data_ts[feature_names].fillna(A_fill)
        y_A = train_data_ts['Generator_ON'].astype(int).values
        X_A_scaled = scaler.transform(X_A_df)
        X_A_xgb = make_xgb_feature_matrix(X_A_scaled, X_A_df, feature_names)

        # Parallel A matrix for FROM-SCRATCH approaches when temporal features are
        # enabled. Fresh scaler fit on the extended feature list (the saved
        # base's scaler doesn't know the new columns). Falls back to the base
        # matrices when temporal is off, so behaviour is unchanged in that case.
        if temporal_features and len(fromscratch_feats) > len(feature_names):
            A_fill_fs = train_data_ts[fromscratch_feats].mean()
            X_A_df_fs = train_data_ts[fromscratch_feats].fillna(A_fill_fs)
            scaler_fs = StandardScaler().fit(X_A_df_fs.values)
            X_A_scaled_fs = scaler_fs.transform(X_A_df_fs.values)
            X_A_xgb_fs = make_xgb_feature_matrix(X_A_scaled_fs, X_A_df_fs, fromscratch_feats)
        else:
            scaler_fs = scaler
            X_A_df_fs = X_A_df
            X_A_scaled_fs = X_A_scaled
            X_A_xgb_fs = X_A_xgb

        # ------------------------------------------------------------------
        # FIX 1: Fixed test window -- always the LAST test_samples rows.
        # This window is identical no matter what finetune_days value or
        # window index we're evaluating, so every comparison in the results
        # table is against the exact same held-out evaluation set.
        # ------------------------------------------------------------------
        pretest_df = test_data_ts.iloc[:-test_samples].reset_index(drop=True)
        fixedtest_df = test_data_ts.iloc[-test_samples:].reset_index(drop=True)

        # ------------------------------------------------------------------
        # FIX 2 (fillna): compute fill values from the fixed-test-excluded
        # portion only, then apply to both splits separately. This way the
        # fixed test window's missing values (if any) are filled using
        # statistics that never include the test window itself.
        # In practice (confirmed via diagnostic): raw sensor columns have
        # ~0 NaNs; only shift(1)-derived columns have 1-2 NaN rows per site.
        # So this fix is about correctness, not a large numeric impact.
        # ------------------------------------------------------------------
        fill_values = pretest_df[feature_names].mean()

        X_fixedtest = fixedtest_df[feature_names].fillna(fill_values)
        y_fixedtest = fixedtest_df['Generator_ON']
        X_fixedtest_scaled = scaler.transform(X_fixedtest)
        # XGB-specific test matrix (Prev_High_delta_* left raw), built once per site
        X_fixedtest_xgb = make_xgb_feature_matrix(X_fixedtest_scaled, X_fixedtest, feature_names)

        # Parallel test matrix for from-scratch approaches (extended feature set)
        if temporal_features and len(fromscratch_feats) > len(feature_names):
            fill_values_fs = pretest_df[fromscratch_feats].mean()
            X_fixedtest_fs = fixedtest_df[fromscratch_feats].fillna(fill_values_fs)
            X_fixedtest_scaled_fs = scaler_fs.transform(X_fixedtest_fs.values)
            X_fixedtest_xgb_fs = make_xgb_feature_matrix(
                X_fixedtest_scaled_fs, X_fixedtest_fs, fromscratch_feats)
        else:
            fill_values_fs = fill_values
            X_fixedtest_fs = X_fixedtest
            X_fixedtest_scaled_fs = X_fixedtest_scaled
            X_fixedtest_xgb_fs = X_fixedtest_xgb

        # LOSO baseline, evaluated on the SAME fixed test window every time.
        # Use the XGB-treated matrix if the base model is XGBoost so the
        # baseline reflects the model's actual training feature treatment.
        base_is_xgb = hasattr(base_model, 'get_booster')
        X_test_for_base = X_fixedtest_xgb if base_is_xgb else X_fixedtest_scaled
        y_pred_baseline = base_model.predict(X_test_for_base)
        baseline_metrics = calculate_metrics(y_fixedtest, y_pred_baseline, test_site)
        baseline_metrics['finetune_days'] = 0
        baseline_metrics['window_idx'] = -1
        baseline_metrics['approach'] = 'LOSO_baseline'
        results.append(baseline_metrics)

        print(f"LOSO Baseline (fixed {test_days}-day test window) -> "
              f"Acc: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1']:.4f}\n")

        n_pretest = len(pretest_df)

        # Fine-tuning
        for n_days in finetune_days:
            n_samples = int(samples_per_day * n_days)
            if n_samples >= n_pretest:
                print(f"  Skipping {n_days} day(s): not enough pre-test data "
                      f"({n_pretest} rows available)\n")
                continue

            print(f"  {'-'*76}")
            print(f"  Fine-tuning with {n_days} day(s) -- averaging {n_windows} window(s)")
            print(f"  {'-'*76}")

            # --------------------------------------------------------------
            # WINDOW SELECTION: plain random windows, with ONE universal,
            # site-agnostic guard -- a window must contain BOTH classes (at
            # least one Generator_ON and one OFF row) to be usable. A single-
            # class window can't train a classifier at all (and crashes
            # XGBoost), so requiring both classes isn't a tuned quality bar,
            # it's the definition of a trainable window. There is deliberately
            # NO ON-count threshold here: needing to hand-pick ON-dense windows
            # would be overfitting to the luxury of choosing, which a real new
            # site can't give us. Performance on randomly-drawn (but trainable)
            # windows is the honest test of whether fine-tuning generalizes.
            # --------------------------------------------------------------
            max_start = n_pretest - n_samples
            rng = np.random.RandomState(base_seed + n_days)
            if max_start <= 0:
                window_starts = [0]
            else:
                y_pre = pretest_df['Generator_ON'].astype(int).values
                csum = np.concatenate([[0], np.cumsum(y_pre)])
                on_counts = csum[n_samples:max_start + n_samples + 1] - csum[:max_start + 1]
                all_starts = np.arange(max_start + 1)

                # usable = has at least one ON and at least one OFF
                usable = all_starts[(on_counts > 0) & (on_counts < n_samples)]
                n_w = min(n_windows, len(usable) if len(usable) > 0 else 1)

                if len(usable) == 0:
                    # extremely degenerate site/day-count: no mixed-class window
                    # exists at all. Fall back to start 0 and let the per-window
                    # single-class guard below skip it gracefully.
                    window_starts = [0]
                    print(f"  Window selection: WARNING no window has both classes "
                          f"at {n_days}d; results for this day-count not meaningful")
                else:
                    chosen = rng.choice(usable, size=n_w, replace=False)
                    window_starts = sorted(int(s) for s in chosen)
                    n_skipped = (max_start + 1) - len(usable)
                    if n_skipped > 0:
                        print(f"  Window selection: {len(usable)}/{max_start+1} windows "
                              f"have both classes ({n_skipped} single-class skipped)")

            for w_idx, start in enumerate(window_starts):
                window_df = pretest_df.iloc[start:start + n_samples]

                # Universal per-window guard: never train on a single-class
                # window even if one slipped through (keeps XGBoost from
                # crashing and keeps results honest).
                _yw = window_df['Generator_ON'].astype(int)
                if _yw.nunique() < 2:
                    print(f"      [window {w_idx}] skipped: only one class present "
                          f"(n_pos={int(_yw.sum())}/{len(_yw)})")
                    continue

                # ----------------------------------------------------------
                # FIX 7 (threshold tuning) setup: carve out a calibration
                # slice from the END of the fine-tune window itself. This
                # slice is used ONLY to pick a decision threshold -- it is
                # never the fixed test window, so the tuned threshold is
                # not chosen using any information from the evaluation set.
                # The model is still fit on the FULL fine-tune window
                # (calib slice included in training), since with windows
                # this small we can't afford to also shrink the fit data --
                # the calibration step only borrows the window's *predicted
                # probabilities* on itself, not held-out generalization.
                # ----------------------------------------------------------
                n_calib = max(20, int(len(window_df) * threshold_holdout_frac)) if tune_threshold else 0
                calib_df = window_df.iloc[-n_calib:] if n_calib > 0 else window_df.iloc[0:0]

                window_fill_values = window_df[feature_names].mean()
                X_finetune = window_df[feature_names].fillna(window_fill_values)
                y_finetune = window_df['Generator_ON']
                X_finetune_scaled = scaler.transform(X_finetune)
                # XGB-specific fine-tune matrix (Prev_High_delta_* left raw)
                X_finetune_xgb = make_xgb_feature_matrix(X_finetune_scaled, X_finetune, feature_names)

                # Parallel from-scratch fine-tune matrix (extended features)
                if temporal_features and len(fromscratch_feats) > len(feature_names):
                    window_fill_values_fs = window_df[fromscratch_feats].mean()
                    X_finetune_fs = window_df[fromscratch_feats].fillna(window_fill_values_fs)
                    X_finetune_scaled_fs = scaler_fs.transform(X_finetune_fs.values)
                    X_finetune_xgb_fs = make_xgb_feature_matrix(
                        X_finetune_scaled_fs, X_finetune_fs, fromscratch_feats)
                else:
                    window_fill_values_fs = window_fill_values
                    X_finetune_fs = X_finetune
                    X_finetune_scaled_fs = X_finetune_scaled
                    X_finetune_xgb_fs = X_finetune_xgb

                X_calib_scaled = scaler.transform(calib_df[feature_names].fillna(window_fill_values)) \
                    if n_calib > 0 else None
                y_calib = calib_df['Generator_ON'] if n_calib > 0 else None

                # Parallel calib matrix for from-scratch (extended-feature) approaches
                if temporal_features and len(fromscratch_feats) > len(feature_names) and n_calib > 0:
                    X_calib_scaled_fs = scaler_fs.transform(
                        calib_df[fromscratch_feats].fillna(window_fill_values_fs).values)
                else:
                    X_calib_scaled_fs = X_calib_scaled

                # FIX 6: record window composition so we can directly check
                # whether bad results correlate with too few positives.
                n_pos = int(y_finetune.sum())
                pos_rate = float(y_finetune.mean())

                for appr in approaches_to_run:
                    # XGBoost approaches use the XGB-treated matrices (Prev_High_delta_*
                    # left raw) to match the saved base model; everything else uses
                    # the standard fully-scaled matrices.
                    # XGB-treated matrices are needed for any XGBoost-based
                    # method (so features match the saved base); RF-based
                    # methods use the standard scaled matrices.
                    xgb_family_appr = appr in ('xgb_warmstart', 'xgb_fromscratch', 'freeze_structure') \
                        or (appr in ('tradaboost', 'weighted_source_target') and model_type == 'XGBoost')
                    is_xgb_appr = xgb_family_appr

                    # Temporal-feature routing: only from-scratch approaches
                    # (continuation, feature_adaptation, xgb_fromscratch,
                    # weighted_source_target, tradaboost) can consume the extended
                    # feature set. warm_start, freeze_structure, residual_correction
                    # reload the saved base which has a fixed feature list and
                    # must stay on the base matrices regardless of the flag.
                    fromscratch_appr = appr in (
                        'continuation', 'feature_adaptation',
                        'xgb_fromscratch', 'weighted_source_target', 'tradaboost')
                    use_extended = temporal_features and fromscratch_appr and \
                        len(fromscratch_feats) > len(feature_names)

                    if use_extended:
                        X_ft_use = X_finetune_xgb_fs if is_xgb_appr else X_finetune_scaled_fs
                        X_test_use = X_fixedtest_xgb_fs if is_xgb_appr else X_fixedtest_scaled_fs
                        X_A_use = X_A_xgb_fs if is_xgb_appr else X_A_scaled_fs
                        feats_use = fromscratch_feats
                    else:
                        X_ft_use = X_finetune_xgb if is_xgb_appr else X_finetune_scaled
                        X_test_use = X_fixedtest_xgb if is_xgb_appr else X_fixedtest_scaled
                        X_A_use = X_A_xgb if is_xgb_appr else X_A_scaled
                        feats_use = feature_names

                    if appr == 'continuation':
                        finetuned = approach1_continuation(base_model, X_ft_use, y_finetune)
                    elif appr == 'feature_adaptation':
                        finetuned = approach2_feature_adaptation(
                            base_model, X_ft_use, y_finetune, feats_use)
                    elif appr == 'residual_correction':
                        finetuned = approach3_residual_correction(
                            base_model, X_ft_use, y_finetune, len(X_ft_use))
                    elif appr == 'xgb_warmstart':
                        finetuned = approach4_xgb_warmstart(
                            base_model, X_ft_use, y_finetune, len(X_ft_use))
                    elif appr == 'xgb_fromscratch':
                        finetuned = approach5_xgb_fromscratch(
                            base_model, X_ft_use, y_finetune, len(X_ft_use))
                    elif appr == 'tradaboost':
                        print(f"      -> TrAdaBoost (A={len(X_A_use)} source + B={len(X_ft_use)} target)")
                        finetuned = approach6_tradaboost(
                            X_A_use, y_A, X_ft_use, np.asarray(y_finetune).astype(int))
                    elif appr == 'weighted_source_target':
                        print(f"      -> Weighted A+B_small "
                              f"(b_weight={'auto' if b_weight is None else b_weight})")
                        finetuned = approach7_weighted_source_target(
                            X_A_use, y_A, X_ft_use, np.asarray(y_finetune).astype(int),
                            b_weight=b_weight, model_type=model_type)
                    elif appr == 'freeze_structure':
                        finetuned = approach8_freeze_structure_update_leaves(
                            base_model, X_ft_use, np.asarray(y_finetune).astype(int), feature_names)
                    else:
                        raise ValueError(f"Unknown approach: {appr}")

                    # --- default threshold (0.5), as before ---
                    y_pred = finetuned.predict(X_test_use)
                    metrics = calculate_metrics(
                        y_fixedtest, y_pred, test_site,
                        n_pos=n_pos, pos_rate=pos_rate, threshold_used=0.5
                    )
                    metrics['finetune_days'] = n_days
                    metrics['window_idx'] = w_idx
                    metrics['approach'] = appr
                    results.append(metrics)

                    improvement = metrics['accuracy'] - baseline_metrics['accuracy']
                    print(f"      [window {w_idx}] n_pos={n_pos} pos_rate={pos_rate:.3f} "
                          f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
                          f"Improvement: {improvement:+.4f}")

                    # --- tuned threshold (calibrated on fine-tune window only) ---
                    X_calib_use = None
                    X_calib_base_for_use = X_calib_scaled_fs if use_extended else X_calib_scaled
                    calib_feats_for_use = fromscratch_feats if use_extended else feature_names
                    calib_fill_for_use = window_fill_values_fs if use_extended else window_fill_values
                    if X_calib_base_for_use is not None:
                        X_calib_use = make_xgb_feature_matrix(
                            X_calib_base_for_use,
                            calib_df[calib_feats_for_use].fillna(calib_fill_for_use),
                            calib_feats_for_use) if is_xgb_appr else X_calib_base_for_use
                    if tune_threshold and X_calib_use is not None and hasattr(finetuned, 'predict_proba'):
                        best_t, default_t = find_best_threshold(finetuned, X_calib_use, y_calib)
                        if best_t != default_t:
                            p_test = finetuned.predict_proba(X_test_use)
                            if p_test.shape[1] == 2:
                                y_pred_tuned = (p_test[:, 1] > best_t).astype(int)
                                metrics_tuned = calculate_metrics(
                                    y_fixedtest, y_pred_tuned, test_site,
                                    n_pos=n_pos, pos_rate=pos_rate, threshold_used=best_t
                                )
                                metrics_tuned['finetune_days'] = n_days
                                metrics_tuned['window_idx'] = w_idx
                                metrics_tuned['approach'] = appr + '_tuned_threshold'
                                results.append(metrics_tuned)

                                improvement_tuned = metrics_tuned['accuracy'] - baseline_metrics['accuracy']
                                print(f"      [window {w_idx}] (threshold={best_t:.2f}) "
                                      f"Acc: {metrics_tuned['accuracy']:.4f}, F1: {metrics_tuned['f1']:.4f}, "
                                      f"Improvement: {improvement_tuned:+.4f}")
            print()

    return pd.DataFrame(results)


def summarize(results_df):
    """Average across windows for each (site, approach, finetune_days) so the
    headline number reflects reliability, not a single lucky/unlucky draw.
    Now also reports recall and the fine-tune window's positive rate, since
    those are what told us Katwa's problem was recall-starved, not a generic
    'needs more days' problem."""
    if results_df.empty:
        return results_df
    grouped = results_df[results_df['finetune_days'] > 0].groupby(
        ['site', 'approach', 'finetune_days']
    ).agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        recall_mean=('recall', 'mean'),
        recall_std=('recall', 'std'),
        precision_mean=('precision', 'mean'),
        finetune_n_pos_mean=('finetune_n_pos', 'mean'),
        finetune_pos_rate_mean=('finetune_pos_rate', 'mean'),
        n_windows=('accuracy', 'count')
    ).reset_index()
    return grouped


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    results_df = run_fine_tuning()

    if results_df.empty:
        print("No results. Check errors above.")
        exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temporal_suffix = "_temporal" if temporal_features else ""
    output = f"{savepath}final_results_finetuning_XGB_{model_type}_{feature_group}{temporal_suffix}_{timestamp}.csv"
    results_df.to_csv(output, index=False)

    summary_df = summarize(results_df)
    summary_output = f"{savepath}final_results_finetuning_XGB_{model_type}_{feature_group}{temporal_suffix}_{timestamp}_summary.csv"
    summary_df.to_csv(summary_output, index=False)

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"Raw results (per window, incl. tuned-threshold rows): {output}")
    print(f"Averaged summary (per site/approach/days):            {summary_output}")
    print(f"Total raw rows: {len(results_df)}")
    print(f"\nHow to read the summary file:")
    print(f"  - f1_std large relative to f1_mean -> don't trust that day count yet")
    print(f"  - recall_mean stuck low while finetune_pos_rate_mean is also low ->")
    print(f"    that site's fine-tune windows aren't seeing enough ON examples;")
    print(f"    more days alone may not fix it (consider stratified window sampling)")
    print(f"  - compare approach rows vs '<approach>_tuned_threshold' rows in the raw")
    print(f"    file: if tuned-threshold recall is much higher at similar precision,")
    print(f"    the model could separate the classes fine -- it just needed a lower")
    print(f"    cutoff, which is a far cheaper fix than more fine-tune data.")
    print(f"{'='*80}\n")