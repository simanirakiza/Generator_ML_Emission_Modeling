"""
step1_temporal_features_test.py
================================

STANDALONE DIAGNOSTIC -- does NOT modify fine_tune_from_saved_models.py or
cross_site_validation.py. Sits alongside them, imports what it can, and runs
the same fine-tuning evaluation with ONE additive change: rolling-window
temporal features on top of the existing group3/group4 features.

WHY THIS EXISTS
---------------
Katwa's failure mode is pointwise-ambiguous / temporally-informative:
  * ~11% of Katwa ON rows sit inside the OFF frequency band (measured earlier)
  * ON episodes drift 49-54 Hz WITHIN an episode and run for hours with a
    distinctive variance signature
So a single 2-minute row at 50.1 Hz is unclassifiable, but a 40-minute
trajectory of drift + wander + variance is not. Every model we run today
sees each row essentially independently (features derived per-timestamp).

BEFORE spending weeks on sequence models or self-supervised pretraining,
this script tests the cheap version of that hypothesis:
    if adding rolling std/range/drift/transition features to XGBoost
    materially improves Katwa recall, temporal modeling is the right axis
    to invest in. If they do nothing, we saved that time.

DESIGN CONSTRAINTS (deliberately chosen)
----------------------------------------
1. Piggyback on the SAVED LOSO base model's feature space via a wrapper:
   we augment the fine-tune / test matrices with new columns, and train
   a fresh XGBoost head from scratch on (source A + new-site B) with
   auto domain balance (b_weight = n_A/n_B) -- i.e. the weighted_source_target
   recipe from approach7, which was the empirically strongest method for
   Katwa. We do NOT reload the LOSO base's XGBoost model here: it can't
   accept new feature columns, so re-using its trees would defeat the
   whole test. The COMPARISON is 'weighted_source_target with vs without
   temporal features on the SAME random both-class windows'.

2. Uses the same random-window / both-classes-only selection rule (no
   ON-density cherry-picking) so results reflect what a real deployment
   week would give.

3. Uses the same LOSO baseline (base model, no fine-tuning) as the point of
   comparison for the "with temporal features" curve.

4. Adds NO tunable knobs. Rolling-window sizes are hardcoded at
   15/30/60-minute equivalents (matching the paper's mean-threshold window
   and Katwa's within-episode drift timescale). Windowing rule and b_weight
   are inherited unchanged.

OUTPUTS
-------
CSV: results_temporal_test_{model}_{group}_{timestamp}.csv
     One row per (site, finetune_days, window, variant) where variant is
     either 'baseline_group' or 'baseline_group+temporal'. Columns include
     f1, precision, recall, accuracy, gen_detection_error, n_pos, pos_rate.

Also prints a per-site summary at the end comparing avg F1 and recall
across variants, so you don't need to open the CSV to see the answer.

USAGE
-----
python step1_temporal_features_test.py --root_path /Users/mahsa/PycharmProjects/ \
    --model_type XGBoost --feature_group group4 --finetune_days "3,5,7" --test_days 12
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime
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


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default="/Users/mahsa/PycharmProjects/")
parser.add_argument('--finetune_days', type=str, default="3,5,7,14,20",
                    help="Comma-separated integers, e.g. '3,5,7'")
parser.add_argument('--test_days', type=int, default=7,
                    help="Fixed last-N-days test window per site (freeze this "
                         "for cross-run comparability)")
parser.add_argument('--n_windows', type=int, default=5)
parser.add_argument('--model_type', type=str, default="XGBoost",
                    choices=["XGBoost"],
                    help="XGBoost only in this diagnostic; the feature-space "
                         "comparison is a like-for-like XGB test.")
parser.add_argument('--feature_group', type=str, default="group4",
                    choices=["group3", "group4"])
parser.add_argument('--sites', type=str, default="Ndosho,Walikale,Kitatumba,Katwa",
                    help="Comma-separated sites to run; default is all four")
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

if not _HAS_XGB:
    raise SystemExit("xgboost is required for this diagnostic. pip install xgboost")

root_path = args.root_path
finetune_days_list = [int(x) for x in args.finetune_days.split(",")]
test_days = args.test_days
n_windows = args.n_windows
feature_group = args.feature_group
model_type = args.model_type
sites = [s.strip() for s in args.sites.split(",")]
base_seed = args.seed

minutes_interval = 2
samples_per_day = 720  # 24*60/2

print(f"\n{'='*80}")
print(f"STEP 1: temporal-features hypothesis test")
print(f"{'='*80}")
print(f"Root:           {root_path}")
print(f"Sites:          {sites}")
print(f"Finetune days:  {finetune_days_list}")
print(f"Test days:      {test_days} (fixed)")
print(f"Windows/value:  {n_windows}")
print(f"Feature group:  {feature_group}  (temporal features are ADDITIVE to this)")
print(f"{'='*80}\n")


# ----------------------------------------------------------------------------
# Data loading -- mirrors fine_tune_from_saved_models.py exactly so the test
# is a like-for-like comparison
# ----------------------------------------------------------------------------
data_paths_all = {
    "Ndosho":    f"{root_path}Generator_ML_Emission_Modeling/data/0. ChCbNdosho/training",
    "Walikale":  f"{root_path}Generator_ML_Emission_Modeling/data/1. Combination 1_Walikale/training",
    "Kitatumba": f"{root_path}Generator_ML_Emission_Modeling/data/5. Combination 5/training/kitatumba",
    "Katwa":     f"{root_path}Generator_ML_Emission_Modeling/data/5. Combination 5/training/katwa",
}

site_columns_oi = {
    "Walikale": ['Frequency 09B35150 (Hgr Walikale - Lab)',
                 'Frequency 64B734C2 (Hgr Walikale - Cabane)',
                 'Voltage 09B35150 (Hgr Walikale - Lab)',
                 'Voltage 64B734C2 (Hgr Walikale - Cabane)'],
    "Ndosho":   ['Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                 'Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)',
                 'Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                 'Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)'],
    "Katwa":    ['Frequency 1C712FB8 (Hgr Katwa - Labo)',
                 'Frequency AEBA416D (Hgr Katwa - Cabine)',
                 'Voltage 1C712FB8 (Hgr Katwa - Labo)',
                 'Voltage AEBA416D (Hgr Katwa - Cabine)'],
    "Kitatumba": ['Frequency 6EF39700 (Hgr Kitatumba - Labo)',
                  'Frequency D2B9CB2E (Hgr Kitatumba - Mdh office)',
                  'Voltage 6EF39700 (Hgr Kitatumba - Labo)',
                  'Voltage D2B9CB2E (Hgr Kitatumba - Mdh office)'],
}

print("Loading data...")
all_sites = list(data_paths_all.keys())
site_dfs = {s: utils.load_data(data_paths_all[s]) for s in all_sites}

def rename_site_columns(df, site):
    freq_cols = site_columns_oi[site][:2]
    volt_cols = site_columns_oi[site][2:]
    rename_map = {freq_cols[0]: 'freq_a', freq_cols[1]: 'freq_b',
                  volt_cols[0]: 'volt_a', volt_cols[1]: 'volt_b'}
    df = df.rename(columns=rename_map)
    df['Site'] = site
    return df

processed = [rename_site_columns(df, s) for s, df in site_dfs.items()]
combined_df = pd.concat(processed, ignore_index=True)
combined_df['Time'] = pd.to_datetime(combined_df['Time'])
print(f"OK loaded {len(combined_df):,} rows across {len(all_sites)} sites\n")


# ----------------------------------------------------------------------------
# TEMPORAL FEATURES: rolling-window summaries of V/F traces
# ----------------------------------------------------------------------------
def add_temporal_features(df_ts, freq_cols=('freq_a', 'freq_b'),
                          volt_cols=('volt_a', 'volt_b')):
    """
    Add rolling-window features to a per-site time-ordered dataframe.
    Windows: 15 / 30 / 60 min (equivalent to 7 / 15 / 30 rows at 2-min sampling).
    All windows are BACKWARD-LOOKING (min_periods=1) so no future information
    leaks into any row. The columns added are:
      * rolling std      (V,F, per sensor, per window)     -- variance signature
      * rolling range    (V,F, per sensor, per window)     -- drift within episode
      * rolling mean-drift = current - rolling_mean(window) -- direction of shift
      * transition count of |delta| > small_eps over window -- how "jumpy"
    These are the pointwise-invisible / episode-visible quantities that
    should discriminate Katwa's dispersed ON state from OFF, if the
    temporal hypothesis is correct.

    Returns a new dataframe with the extra columns; original data untouched.
    """
    df = df_ts.copy()

    # do it per-site so a rolling window never spans two sites
    if 'Site' not in df.columns:
        raise ValueError("dataframe must have a 'Site' column to avoid "
                         "rolling across sites")

    windows_rows = {'15min': 7, '30min': 15, '60min': 30}
    eps_freq, eps_volt = 0.5, 5.0   # "meaningful jump" thresholds

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


# ----------------------------------------------------------------------------
# Feature-pipeline helpers (mirror fine_tune_from_saved_models.py)
# ----------------------------------------------------------------------------
XGB_RAW_COLS = ['Prev_High_delta_freq_lab', 'Prev_High_delta_freq_clinique',
                'Prev_High_delta_volt_lab', 'Prev_High_delta_volt_clinique']

def make_xgb_matrix(X_scaled, X_raw_df, feature_names):
    """Same treatment as in the main script: Prev_High_delta_* stay RAW
    (unscaled) so they match how the XGBoost pipeline was designed."""
    X = X_scaled.copy()
    for col in XGB_RAW_COLS:
        if col in feature_names:
            idx = feature_names.index(col)
            X[:, idx] = pd.to_numeric(X_raw_df[col], errors='coerce').values
    return X


def build_features(other_sites_data, held_out_data, feature_group):
    """Run the same feature pipeline as the main script, so the ONLY delta
    in this diagnostic is the additive temporal columns."""
    columns_to_compare = [('freq_a', 'freq_lab'), ('freq_b', 'freq_clinique'),
                          ('volt_a', 'volt_lab'), ('volt_b', 'volt_clinique')]
    train_ts, test_ts = utils.generate_group3features(
        other_sites_data, held_out_data, columns_to_compare,
        'freq_a', 'freq_b', 'volt_a', 'volt_b')
    if feature_group == "group4":
        train_ts, test_ts, hmm_features = utils.generate_group4features(
            train_ts, test_ts, 'freq_a', 'freq_b', 'volt_a', 'volt_b')
    return train_ts, test_ts


def train_weighted_xgb(X_A, y_A, X_B, y_B, seed=42):
    """The empirically-strongest approach from earlier runs:
    weighted_source_target with auto domain balance (b_weight = n_A/n_B),
    XGBoost, from-scratch head. This is what we're comparing WITH vs
    WITHOUT the temporal features."""
    b_weight = len(X_A) / max(len(X_B), 1)
    X = np.vstack([X_A, X_B])
    y = np.concatenate([y_A, y_B]).astype(int)
    sw = np.ones(len(X))
    sw[len(X_A):] = b_weight
    npos = int(np.sum(y)); nneg = len(y) - npos
    spw = (nneg / npos) if npos > 0 else 1.0
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                      random_state=seed, eval_metric='logloss',
                      scale_pos_weight=spw)
    m.fit(X, y, sample_weight=sw)
    return m, b_weight


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
    }
    n_true_pos = int(y_true.sum())
    n_pred_pos = int(y_pred.sum())
    m['gen_detection_error'] = (abs(n_pred_pos - n_true_pos) / n_true_pos * 100
                                if n_true_pos > 0 else 0.0)
    return m


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def run():
    results = []
    savepath = f"{root_path}Generator_ML_Emission_Modeling/Modeling/summer25_results/cross_site_models/"

    for test_site in sites:
        if test_site not in all_sites:
            print(f"! Unknown site '{test_site}', skipping")
            continue
        print(f"\n{'='*80}\nSITE: {test_site}\n{'='*80}\n")

        # LOSO base model (used ONLY for the baseline-no-finetune reference row
        # so we can plot the same day-0 point as the main experiments)
        model_filename = f"loso_baseline_{test_site}_{model_type}_{feature_group}.pkl"
        model_path = os.path.join(savepath, "saved_models", model_filename)
        if not os.path.exists(model_path):
            print(f"  X Missing base model {model_filename}; skipping site\n")
            continue
        saved = joblib.load(model_path)
        base_model, base_scaler, base_feat_names = (
            saved['model'], saved['scaler'], saved['feature_names'])

        # Split data
        held_out = combined_df[combined_df['Site'] == test_site]\
            .sort_values('Time').reset_index(drop=True)
        other_sites = combined_df[combined_df['Site'] != test_site].copy()

        # Full base-group feature build (matches main script)
        train_ts, test_ts_full = build_features(other_sites, held_out, feature_group)

        # Same "last test_days days" holdout policy as main script
        test_samples = int(samples_per_day * test_days)
        if len(test_ts_full) <= test_samples:
            print(f"  Skipping {test_site}: not enough held-out data\n")
            continue
        pretest = test_ts_full.iloc[:-test_samples].reset_index(drop=True)
        fixedtest = test_ts_full.iloc[-test_samples:].reset_index(drop=True)

        # LOSO baseline on fixed test window (as reference; matches main script's
        # LOSO_baseline row). Uses the SAVED model's feature list.
        fill = pretest[base_feat_names].mean()
        X_test_base = fixedtest[base_feat_names].fillna(fill)
        X_test_base_s = base_scaler.transform(X_test_base)
        X_test_base_xgb = make_xgb_matrix(X_test_base_s, X_test_base, base_feat_names)
        y_test = fixedtest['Generator_ON'].astype(int).values
        loso_pred = base_model.predict(X_test_base_xgb)
        m = score(y_test, loso_pred)
        m.update({'site': test_site, 'finetune_days': 0, 'window_idx': -1,
                  'variant': 'LOSO_baseline', 'n_pos_finetune': 0, 'pos_rate': 0.0})
        results.append(m)
        print(f"  LOSO_baseline (base group only): F1={m['f1']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f}")

        # --------------------------------------------------------------
        # Build the AUGMENTED feature space: base features + temporal columns.
        # Temporal features must be computed on the ORIGINAL time series
        # (raw freq/volt), then merged in -- so we regenerate on the raw
        # 4-column dataframes and concatenate.
        # --------------------------------------------------------------
        # The main pipeline works on already-featurized outputs; we need to
        # rebuild the underlying 4-column series for both train and test to
        # compute rolling windows correctly. We keep the same generate_group*
        # pipeline output and then splice temporal columns in by Time+Site.
        raw_source = other_sites[['Time', 'Site', 'freq_a', 'freq_b',
                                  'volt_a', 'volt_b']].copy()
        raw_target = held_out[['Time', 'Site', 'freq_a', 'freq_b',
                               'volt_a', 'volt_b']].copy()
        raw_all = pd.concat([raw_source, raw_target], ignore_index=True)
        raw_all, temporal_cols = add_temporal_features(raw_all)

        # Merge temporal columns back onto the featurized frames on (Time,Site)
        # if 'Site' column present, else on Time only (defensive).
        def splice(featurized_df):
            merge_cols = ['Time'] + (['Site'] if 'Site' in featurized_df.columns else [])
            temp_df = raw_all[merge_cols + temporal_cols].drop_duplicates(subset=merge_cols)
            return featurized_df.merge(temp_df, on=merge_cols, how='left')

        train_ts_aug = splice(train_ts)
        pretest_aug = splice(pretest)
        fixedtest_aug = splice(fixedtest)

        aug_feat_names = list(base_feat_names) + [c for c in temporal_cols
                                                   if c in train_ts_aug.columns]
        print(f"  Augmented feature count: {len(aug_feat_names)} "
              f"(base {len(base_feat_names)} + temporal "
              f"{len(aug_feat_names) - len(base_feat_names)})")

        # Build A (source) matrices ONCE per site, both variants
        # BASE
        X_A_base_df = train_ts[base_feat_names].fillna(train_ts[base_feat_names].mean())
        y_A = train_ts['Generator_ON'].astype(int).values
        scaler_base = StandardScaler().fit(X_A_base_df.values)
        X_A_base_s = scaler_base.transform(X_A_base_df.values)
        X_A_base_xgb = make_xgb_matrix(X_A_base_s, X_A_base_df, base_feat_names)
        # AUGMENTED
        X_A_aug_df = train_ts_aug[aug_feat_names].fillna(
            train_ts_aug[aug_feat_names].mean())
        scaler_aug = StandardScaler().fit(X_A_aug_df.values)
        X_A_aug_s = scaler_aug.transform(X_A_aug_df.values)
        X_A_aug_xgb = make_xgb_matrix(X_A_aug_s, X_A_aug_df, aug_feat_names)

        # Fixed-test matrices (both variants)
        Xt_base_df = fixedtest[base_feat_names].fillna(fill)
        Xt_base_s = scaler_base.transform(Xt_base_df.values)
        Xt_base_xgb = make_xgb_matrix(Xt_base_s, Xt_base_df, base_feat_names)

        fill_aug = pretest_aug[aug_feat_names].mean()
        Xt_aug_df = fixedtest_aug[aug_feat_names].fillna(fill_aug)
        Xt_aug_s = scaler_aug.transform(Xt_aug_df.values)
        Xt_aug_xgb = make_xgb_matrix(Xt_aug_s, Xt_aug_df, aug_feat_names)

        n_pretest = len(pretest_aug)

        # ---- fine-tune sweep ----
        for n_days in finetune_days_list:
            n_samples = int(samples_per_day * n_days)
            if n_samples >= n_pretest:
                print(f"  {n_days}d exceeds pretest pool; skipping")
                continue

            max_start = n_pretest - n_samples
            rng = np.random.RandomState(base_seed + n_days)
            y_pre = pretest['Generator_ON'].astype(int).values
            csum = np.concatenate([[0], np.cumsum(y_pre)])
            on_counts = csum[n_samples:max_start + n_samples + 1] - csum[:max_start + 1]
            all_starts = np.arange(max_start + 1)
            usable = all_starts[(on_counts > 0) & (on_counts < n_samples)]
            if len(usable) == 0:
                print(f"  {n_days}d: no both-class window available; skipping")
                continue
            n_w = min(n_windows, len(usable))
            window_starts = sorted(int(s) for s in rng.choice(usable, size=n_w,
                                                              replace=False))

            print(f"\n  {n_days}-day windows: {len(usable)}/{max_start+1} usable, "
                  f"sampled {n_w}")

            for w_idx, start in enumerate(window_starts):
                sl = slice(start, start + n_samples)
                n_pos = int(y_pre[sl].sum())
                pos_rate = n_pos / n_samples
                y_B = y_pre[sl]

                # --- BASE variant fine-tune matrix
                XB_base_df = pretest[base_feat_names].iloc[sl].fillna(fill)
                XB_base_s = scaler_base.transform(XB_base_df.values)
                XB_base_xgb = make_xgb_matrix(XB_base_s, XB_base_df, base_feat_names)
                # --- AUG variant fine-tune matrix
                XB_aug_df = pretest_aug[aug_feat_names].iloc[sl].fillna(fill_aug)
                XB_aug_s = scaler_aug.transform(XB_aug_df.values)
                XB_aug_xgb = make_xgb_matrix(XB_aug_s, XB_aug_df, aug_feat_names)

                variant_scores = {}
                for variant, X_A, X_B, X_te in [
                    ('baseline_group',           X_A_base_xgb, XB_base_xgb, Xt_base_xgb),
                    ('baseline_group+temporal',  X_A_aug_xgb,  XB_aug_xgb,  Xt_aug_xgb),
                ]:
                    m_fit, bw = train_weighted_xgb(X_A, y_A, X_B, y_B,
                                                    seed=42 + w_idx)
                    yhat = m_fit.predict(X_te)
                    sc = score(y_test, yhat)
                    sc.update({'site': test_site, 'finetune_days': n_days,
                               'window_idx': w_idx, 'variant': variant,
                               'n_pos_finetune': n_pos, 'pos_rate': pos_rate,
                               'b_weight_auto': bw})
                    results.append(sc)
                    variant_scores[variant] = sc

                f1_base = variant_scores['baseline_group']['f1']
                f1_temp = variant_scores['baseline_group+temporal']['f1']
                r_base  = variant_scores['baseline_group']['recall']
                r_temp  = variant_scores['baseline_group+temporal']['recall']
                print(f"    [w{w_idx}] n_pos={n_pos:>4} | "
                      f"BASE F1={f1_base:.3f} R={r_base:.3f} | "
                      f"BASE+TEMP F1={f1_temp:.3f} R={r_temp:.3f} | "
                      f"dF1={f1_temp - f1_base:+.3f}")

    return pd.DataFrame(results)


def summarize(df):
    if df.empty:
        return df
    ft = df[df['finetune_days'] > 0]
    grouped = ft.groupby(['site', 'variant', 'finetune_days']).agg(
        f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        recall_mean=('recall', 'mean'), precision_mean=('precision', 'mean'),
        gen_err_mean=('gen_detection_error', 'mean'),
        n_windows=('f1', 'count'),
        n_pos_mean=('n_pos_finetune', 'mean')).reset_index()
    base = df[df['variant'] == 'LOSO_baseline']
    if not base.empty:
        b = base.groupby(['site', 'variant', 'finetune_days']).agg(
            f1_mean=('f1', 'mean'), recall_mean=('recall', 'mean'),
            precision_mean=('precision', 'mean'),
            gen_err_mean=('gen_detection_error', 'mean')).reset_index()
        grouped = pd.concat([b, grouped], ignore_index=True)
    return grouped.sort_values(['site', 'finetune_days', 'variant']).reset_index(drop=True)


if __name__ == "__main__":
    df = run()
    if df.empty:
        print("\nNo results produced. Check errors above.")
        raise SystemExit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    savepath = f"{root_path}Generator_ML_Emission_Modeling/Modeling/summer25_results/cross_site_models/"
    out_raw = os.path.join(savepath, f"results_temporal_test_{model_type}_{feature_group}_{ts}.csv")
    out_sum = os.path.join(savepath, f"results_temporal_test_{model_type}_{feature_group}_{ts}_summary.csv")
    df.to_csv(out_raw, index=False)
    summarize(df).to_csv(out_sum, index=False)
    print(f"\nSaved raw:     {out_raw}")
    print(f"Saved summary: {out_sum}")

    # Pretty per-site verdict
    print(f"\n{'='*80}\nSTEP 1 VERDICT (avg F1 across windows, weighted_source_target)")
    print(f"{'='*80}")
    ft = df[df['finetune_days'] > 0]
    for site in sorted(ft['site'].unique()):
        print(f"\n{site}:")
        s = ft[ft['site'] == site]
        for d in sorted(s['finetune_days'].unique()):
            b = s[(s['finetune_days'] == d) & (s['variant'] == 'baseline_group')]
            a = s[(s['finetune_days'] == d) & (s['variant'] == 'baseline_group+temporal')]
            if len(b) and len(a):
                delta_f1 = a['f1'].mean() - b['f1'].mean()
                delta_r  = a['recall'].mean() - b['recall'].mean()
                arrow = "^" if delta_f1 > 0.01 else ("v" if delta_f1 < -0.01 else "-")
                print(f"  {d}d: BASE F1={b['f1'].mean():.3f} R={b['recall'].mean():.3f} "
                      f"| +TEMP F1={a['f1'].mean():.3f} R={a['recall'].mean():.3f} "
                      f"| dF1={delta_f1:+.3f} dR={delta_r:+.3f} {arrow}")
    print(f"\nInterpretation guide:")
    print("  +dF1 >= 0.02 on Katwa AND recall lift => temporal hypothesis supported;")
    print("       proceed to step 2 (self-supervised pretraining).")
    print("  no meaningful change or losses on non-Katwa sites (like HMM did in the")
    print("       IEEE paper for Walikale/Kitatumba) => temporal features are site-")
    print("       specific; consider per-site enabling or drop.")
