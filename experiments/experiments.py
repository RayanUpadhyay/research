"""
Companion experiments for "Fairness and Dependency Preservation in Synthetic Credit Data".

These are NEW runs, not the published results. They follow the paper's design where it is specified
and make each unspecified choice explicit (see DEFINITIONS below). Every number this writes comes
from actually running the code; nothing is filled in by hand.

Run:  mlenv/bin/python experiments.py [n_seeds]
Writes results/raw.jsonl (one line per seed x method) and results/summary.json.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, chi2_contingency, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
EPOCHS, BATCH = 300, 500  # the paper's CTGAN/TVAE settings

DEFINITIONS = {
    'data': 'German Credit: Hofmann, H. (1994), Statlog (German Credit Data), UCI Machine Learning Repository, https://doi.org/10.24432/C5NC77, licensed CC BY 4.0. 1,000 rows, used via a public CSV copy with readable category labels (github.com/selva86/datasets).',
    'target': 'bad credit = 1, good credit = 0 (as in the paper).',
    'split': 'stratified 80/20 train/test per seed; generators see only the training split.',
    'model': 'XGBoost with library defaults (the paper used AutoGluon, which selected XGBoost).',
    'balancing': 'SMOTE, CTGAN, TVAE and Gaussian Copula add synthetic bad-credit rows to the real training set until the classes are 50/50.',
    'protected': 'sex, derived from personal_status_sex: female = "female : divorced/separated/married", male = every other value (the paper notes this attribute is composite). Age group (25 or under) is reported as an extra attribute.',
    'approval': 'predicted good credit (probability of bad credit below 0.5).',
    'disparate_impact': 'approval rate of women divided by approval rate of men (closer to 1 is fairer).',
    'equalized_odds': 'the larger of the true positive rate gap and the false positive rate gap between groups (lower is fairer).',
    'predictive_parity': 'absolute gap in precision of approvals between groups (lower is fairer).',
    'dps': 'numerical part: 1 minus the mean absolute difference between the Spearman correlations of each numeric pair in real vs synthetic rows; categorical part: the same with Cramer\'s V for each categorical pair; DPS = average of the two. 1 means identical dependencies. This is our reading of the paper\'s description, not its exact code.',
    'frobenius': 'Frobenius norm of (real Spearman matrix minus synthetic Spearman matrix) over the numeric columns; lower is better.',
    'smote': 'SMOTE runs on the one-hot data without scaling (as in the paper), so loan amount, in the thousands, dominates which neighbours it picks; SMOTE-NC uses the same numeric distances, so the two produce near-identical numeric values.',
    'validity': 'a run counts as degenerate when any generated column other than the requested label never varies.',
    'seeds': 'every method ran on the same 20 train/test splits (seeds 0-19), so the significance tests are paired.',
    'dcr': 'distance to closest record: for each synthetic row, the distance to its nearest real training row in standardized one-hot space. Compared with the same distance for real held-out rows.',
}

# ---------- data ----------
raw = pd.read_csv(os.path.join(HERE, 'data', 'GermanCredit.csv'))
raw['bad'] = 1 - raw['credit_risk'].astype(int)
raw = raw.drop(columns=['credit_risk'])
CAT = [c for c in raw.columns if raw[c].dtype == object]
NUM = [c for c in raw.columns if c not in CAT and c != 'bad']
sex = np.where(raw['personal_status_sex'] == 'female : divorced/separated/married', 'female', 'male')
young = np.where(raw['age'] <= 25, 'young', 'older')

def encode(df, cols=None):
    X = pd.get_dummies(df.drop(columns=['bad']), columns=CAT, dtype=float)
    if cols is not None: X = X.reindex(columns=cols, fill_value=0.0)
    return X

def decode_onehot(X, cols):
    """turn (possibly fractional) one-hot columns back into categories, for dependency checks"""
    out = pd.DataFrame(index=range(len(X)))
    Xd = pd.DataFrame(X, columns=cols)
    for c in NUM: out[c] = Xd[c].values
    for c in CAT:
        sub = [k for k in cols if k.startswith(c + '_')]
        out[c] = [s[len(c) + 1:] for s in Xd[sub].idxmax(axis=1)]
    return out

# ---------- metrics ----------
def group_rates(y, p, g, a):
    m = g == a; yy, pp = y[m], p[m]
    appr = pp == 0                      # approved = predicted good credit
    tpr = (pp[yy == 1] == 1).mean() if (yy == 1).any() else np.nan   # bad credit correctly caught
    fpr = (pp[yy == 0] == 1).mean() if (yy == 0).any() else np.nan   # good credit wrongly refused
    prec = (yy[appr] == 0).mean() if appr.any() else np.nan          # approvals that were good credit
    return appr.mean(), tpr, fpr, prec

def fairness(y, p, g, prot, ref):
    a1, t1, f1, p1 = group_rates(y, p, g, prot); a0, t0, f0, p0 = group_rates(y, p, g, ref)
    return {'di': a1 / a0 if a0 else np.nan, 'eo': np.nanmax([abs(t1 - t0), abs(f1 - f0)]), 'pp': abs(p1 - p0)}

def cramers_v(a, b):
    t = pd.crosstab(a, b)
    if t.shape[0] < 2 or t.shape[1] < 2: return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]; n = t.values.sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))

def dependency(real, syn):
    sr = spearmanr(real[NUM].astype(float)).correlation
    ss = spearmanr(syn[NUM].astype(float)).correlation
    ss = np.nan_to_num(ss)
    iu = np.triu_indices(len(NUM), 1)
    num = 1 - np.mean(np.abs(sr[iu] - ss[iu]))
    diffs = []
    for i in range(len(CAT)):
        for j in range(i + 1, len(CAT)):
            diffs.append(abs(cramers_v(real[CAT[i]], real[CAT[j]]) - cramers_v(syn[CAT[i]], syn[CAT[j]])))
    cat = 1 - float(np.mean(diffs))
    return {'dps': (num + cat) / 2, 'dps_num': num, 'dps_cat': cat, 'frob': float(np.linalg.norm(sr - ss))}

def valid(df):
    """the checks behind the paper's NaNs: no constant columns, every category of the protected attribute present, no NaNs"""
    probs = []
    if df.isna().any().any(): probs.append('missing values')
    const = [c for c in df.columns if c != 'bad' and df[c].nunique() < 2]   # 'bad' is constant by design: only bad-credit rows were requested
    if const: probs.append('constant columns: ' + ', '.join(const))
    return probs

# ---------- generators ----------
def sdv_synth(kind, train_df, seed):
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    md = Metadata.detect_from_dataframes({'credit': train_df})
    md.update_column('bad', table_name='credit', sdtype='categorical')
    if kind == 'ctgan': s = CTGANSynthesizer(md, epochs=EPOCHS, batch_size=BATCH, verbose=False)
    elif kind == 'tvae': s = TVAESynthesizer(md, epochs=EPOCHS, batch_size=BATCH)
    else: s = GaussianCopulaSynthesizer(md)
    s.fit({'credit': train_df})   # SDV 2 takes and returns a dict of tables
    return s

def sample_rows(s, n, bad=None, seed=0):
    from sdv.sampling import Condition
    np.random.seed(seed)
    out = s.sample(num_rows=n) if bad is None else s.sample_from_conditions([Condition(num_rows=n, column_values={'bad': bad})])
    out = out['credit'] if isinstance(out, dict) else out
    return out.reset_index(drop=True)

def privacy(train_X, syn_X, test_X):
    sc = StandardScaler().fit(train_X)
    nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(train_X))
    d_syn = nn.kneighbors(sc.transform(syn_X))[0][:, 0]
    d_real = nn.kneighbors(sc.transform(test_X))[0][:, 0]
    return {'dcr_syn_median': float(np.median(d_syn)), 'dcr_real_median': float(np.median(d_real)),
            'exact_copies': float(np.mean(d_syn < 1e-6))}

# ---------- one seed ----------
def fit_eval(Xtr, ytr, Xte, yte, gte_sex, gte_age, seed, weights=None, spw=None):
    clf = XGBClassifier(random_state=seed, n_jobs=4, eval_metric='logloss', scale_pos_weight=spw or 1.0)
    # plain arrays: one-hot column names contain characters XGBoost rejects
    clf.fit(np.asarray(Xtr, dtype=float), np.asarray(ytr), sample_weight=weights)
    prob = clf.predict_proba(np.asarray(Xte, dtype=float))[:, 1]; pred = (prob >= 0.5).astype(int)
    r = {'auc': float(roc_auc_score(yte, prob))}
    for k, v in fairness(yte, pred, gte_sex, 'female', 'male').items(): r[k] = float(v)
    for k, v in fairness(yte, pred, gte_age, 'young', 'older').items(): r[k + '_age'] = float(v)
    return r

def reweigh(y, g):
    """Kamiran & Calders reweighing: weight = P(group) * P(label) / P(group, label)"""
    w = np.ones(len(y))
    for a in np.unique(g):
        for l in (0, 1):
            m = (g == a) & (y == l)
            if m.any(): w[m] = (np.mean(g == a) * np.mean(y == l)) / np.mean(m)
    return w

def run_seed(seed, fh):
    idx = np.arange(len(raw))
    tr, te = train_test_split(idx, test_size=0.2, stratify=raw['bad'], random_state=seed)
    tr_df, te_df = raw.iloc[tr].reset_index(drop=True), raw.iloc[te].reset_index(drop=True)
    Xtr = encode(tr_df); cols = list(Xtr.columns); Xte = encode(te_df, cols)
    ytr, yte = tr_df['bad'].values, te_df['bad'].values
    gs, ga = sex[te], young[te]
    n_need = int((ytr == 0).sum() - (ytr == 1).sum())   # synthetic bad-credit rows to reach 50/50

    def emit(method, res, extra=None):
        row = {'seed': seed, 'method': method, **res, **(extra or {})}
        fh.write(json.dumps(row) + '\n'); fh.flush()

    # baselines
    emit('baseline', fit_eval(Xtr, ytr, Xte, yte, gs, ga, seed), {'dps': 1.0, 'frob': 0.0})
    emit('baseline_weighted', fit_eval(Xtr, ytr, Xte, yte, gs, ga, seed, spw=(ytr == 0).sum() / (ytr == 1).sum()))
    Xo, yo = RandomOverSampler(random_state=seed).fit_resample(Xtr, ytr)
    emit('baseline_oversampled', fit_eval(Xo, yo, Xte, yte, gs, ga, seed))
    emit('reweighing', fit_eval(Xtr, ytr, Xte, yte, gs, ga, seed, weights=reweigh(ytr, sex[tr])))

    # SMOTE on one-hot data (as in the paper) and SMOTENC (categorical-aware)
    Xs, ys = SMOTE(random_state=seed, k_neighbors=5).fit_resample(Xtr, ytr)
    new = Xs.iloc[len(Xtr):].values
    syn_dec = decode_onehot(new, cols)
    emit('smote', fit_eval(Xs, ys, Xte, yte, gs, ga, seed), {**dependency(tr_df, syn_dec), **privacy(Xtr.values, new, Xte.values)})
    cat_idx = [i for i, c in enumerate(tr_df.drop(columns=['bad']).columns) if c in CAT]
    Xn, yn = SMOTENC(categorical_features=cat_idx, random_state=seed).fit_resample(tr_df.drop(columns=['bad']), ytr)
    newn = Xn.iloc[len(tr_df):].reset_index(drop=True)
    Xn_enc = encode(pd.concat([Xn, pd.Series(yn, name='bad')], axis=1), cols)
    emit('smotenc', fit_eval(Xn_enc, yn, Xte, yte, gs, ga, seed), {**dependency(tr_df, newn), **privacy(Xtr.values, encode(newn.assign(bad=1), cols).values, Xte.values)})

    # deep and statistical generators
    for kind in ('ctgan', 'tvae', 'gaussian_copula'):
        t0 = time.time()
        try:
            s = sdv_synth(kind if kind != 'gaussian_copula' else 'gc', tr_df, seed)
            add = sample_rows(s, n_need, bad=1, seed=seed)
            probs = valid(add)
            full = pd.concat([tr_df, add], ignore_index=True)
            Xa = encode(full, cols)
            res = fit_eval(Xa, full['bad'].values, Xte, yte, gs, ga, seed)
            uncond = sample_rows(s, len(tr_df), seed=seed)
            extra = {**dependency(tr_df, uncond), **privacy(Xtr.values, encode(uncond, cols).values, Xte.values),
                     'problems': probs, 'fit_seconds': round(time.time() - t0, 1)}
            emit(kind, res, extra)
            # mix experiment: replace a fraction of the real training rows with synthetic rows
            for f in (0.25, 0.5, 0.75, 1.0):
                k = int(round(f * len(tr_df)))
                keep = tr_df.sample(n=len(tr_df) - k, random_state=seed)
                mix = pd.concat([keep, uncond.iloc[:k]], ignore_index=True)
                if mix['bad'].nunique() < 2: continue
                emit(f'mix_{kind}_{int(f * 100)}', fit_eval(encode(mix, cols), mix['bad'].values, Xte, yte, gs, ga, seed))
        except Exception as e:
            emit(kind, {'error': f'{type(e).__name__}: {e}'[:300]})

def summarise():
    rows = [json.loads(l) for l in open(os.path.join(OUT, 'raw.jsonl'))]
    df = pd.DataFrame(rows)
    keys = ['auc', 'di', 'eo', 'pp', 'di_age', 'eo_age', 'pp_age', 'dps', 'dps_num', 'dps_cat', 'frob', 'dcr_syn_median', 'dcr_real_median', 'exact_copies', 'fit_seconds']
    out = {'definitions': DEFINITIONS, 'n_seeds': int(df['seed'].nunique()), 'methods': {}}
    base = df[df.method == 'baseline'].set_index('seed')
    for m, g in df.groupby('method'):
        d = {'n': int(len(g))}
        if 'error' in g and g['error'].notna().any(): d['errors'] = g['error'].dropna().unique().tolist()
        if 'problems' in g: d['problems'] = sorted({p for ps in g['problems'].dropna() for p in ps})
        for k in keys:
            if k in g and g[k].notna().any():
                v = g[k].dropna(); d[k] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
        # paired test against the original-data baseline
        if m != 'baseline':
            gg = g.set_index('seed')
            for k in ('auc', 'di', 'eo', 'pp'):
                if k in gg and gg[k].notna().sum() >= 6:
                    common = gg.index.intersection(base.index)
                    a, b = gg.loc[common, k].values, base.loc[common, k].values
                    try: d.setdefault('p_vs_baseline', {})[k] = float(wilcoxon(a, b).pvalue)
                    except ValueError: pass
        out['methods'][m] = d
    json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    return out

if __name__ == '__main__':
    if sys.argv[-1] == 'summary': print(json.dumps(summarise()['methods'], indent=1)[:3000]); sys.exit()
    with open(os.path.join(OUT, 'raw.jsonl'), 'w') as fh:
        for seed in range(N_SEEDS):
            t = time.time(); run_seed(seed, fh); print(f'seed {seed} done in {time.time() - t:.0f}s', flush=True)
    summarise(); print('ALL DONE')
