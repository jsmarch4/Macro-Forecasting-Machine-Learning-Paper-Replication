"""Complexity mapping following Domenico's train_script.py/test_script.py.

For each family and quantile, fit ALL available architecture x 41-lambda
candidates on the validation period; measure variance of the final
validation-trained model's in-sample predictions; sort by lambda and
select the nearest fitted variance to r times the first lambda=0 variance.
The first reference is recorded explicitly (the original code's tie rule
among lambda=0 architectures is filesystem-order dependent).

The 04/05 validation searches supply architecture lists only; they do not
preselect the architecture. Cached results and verified complexity tables
are not overwritten: outputs go to results/complexity_cross_architecture.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch

from data_utils import load_replication_data, standardize_train_forecast, standardize_target_train_forecast
from models import QuantileNetwork
from losses import pinball_loss_numpy

ROOT = Path('results/complexity')
TAUS = (.05, .10, .25, .50, .75, .90, .95)
FAMILIES = ('linear_activation', 'dnn')
R_GRID = np.round(np.arange(.1, 1.01, .1), 1)
START_VAL, END_VAL = '1980-02-01', '2000-01-01'
START_TEST, END_TEST = '2000-02-01', '2019-12-01'
SEED, LR, EPOCHS_FIRST, EPOCHS_NEXT = 123, .001, 500, 100
CACHE_VERSION = 'domenico_grid41_all_arch_v1'
DEVICE = torch.device('cpu')
torch.set_num_threads(1)


def directory(*parts):
    p = ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def spec_id(spec):
    raw = json.dumps(spec, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def search_architectures():
    """Enumerate ALL saved 04/05 architectures for each family and quantile.

    Validation losses are retained for auditing, not used to preselect an
    architecture. The input row order is retained to define a reproducible
    first-candidate reference at lambda=0 (see build_map).
    """
    architectures = []
    for family in FAMILIES:
        for tau in TAUS:
            path = Path('results') / family / 'search' / f'q{tau:.2f}_search.csv'
            if not path.exists():
                raise FileNotFoundError(f'Missing validation search: {path}')
            df = pd.read_csv(path)
            needed = {'nonlinear_layers', 'hidden_dim', 'alpha', 'lambda', 'validation_loss'}
            if not needed.issubset(df.columns):
                raise ValueError(f'{path}: missing {needed - set(df.columns)}')
            df = df.dropna(subset=['validation_loss'])
            if df.empty:
                raise ValueError(f'No valid search rows: {path}')
            for (layers, dim, alpha), group in df.groupby(
                ['nonlinear_layers', 'hidden_dim', 'alpha'], sort=False
            ):
                # Original train_script.py: linear has 0 layers; DNN has 1 or 2.
                if family == 'linear_activation' and int(layers) != 0:
                    continue
                if family == 'dnn' and int(layers) not in (1, 2):
                    continue
                best = group.loc[group.validation_loss.idxmin()]
                architectures.append(dict(
                    family=family, tau=float(tau),
                    nonlinear_layers=int(layers), hidden_dim=int(dim),
                    alpha=float(alpha),
                    best_validation_lambda=float(best['lambda']),
                    best_validation_loss=float(best.validation_loss),
                ))
    result = pd.DataFrame(architectures)
    result.to_csv(directory('mappings') / 'available_architectures.csv', index=False)
    print(f'Enumerated {len(result)} family/quantile/architecture combinations', flush=True)
    return result


def prepare_cache(X, y, first, last):
    """Construct expanding-window tensors once, not once per candidate."""
    cache = []
    for date in y.loc[first:last].index:
        history_x = X.loc[:date].iloc[:-1]
        history_y = y.loc[:date].iloc[:-1]
        xs, xf = standardize_train_forecast(history_x, X.loc[[date]])
        ys, _, mean_y, _ = standardize_target_train_forecast(history_y, float(y.loc[date]))
        cache.append((date, torch.as_tensor(xs.to_numpy(), dtype=torch.float32),
                      torch.as_tensor(ys.to_numpy(), dtype=torch.float32),
                      torch.as_tensor(xf.to_numpy(), dtype=torch.float32),
                      float(mean_y), float(y.loc[date])))
    return cache


def make_model(spec, n_features):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return QuantileNetwork(n_features, int(spec['nonlinear_layers']),
                           int(spec['hidden_dim']), float(spec['alpha'])).to(DEVICE)


def train_model(model, xt, yt, tau, lam, epochs, lr, optimizer=None):
    """fred.py objective: mean pinball loss + lambda * squared weights."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(xt).reshape(-1)
        if pred.shape != yt.shape:
            raise ValueError(f'Prediction/target shape mismatch: {pred.shape} vs {yt.shape}')
        err = yt - pred
        pinball = torch.maximum(tau * err, (tau - 1) * err).mean()
        penalty = model.l2_penalty()
        (pinball + lam * penalty).backward()
        optimizer.step()
    return model, optimizer


def fit_validation(spec, cache, return_forecasts=False):
    """Warm-start weights and Adam across months; restart Adam for test."""
    model = make_model(spec, cache[0][1].shape[1])
    optimizer = None
    rows = []
    for i, (date, xt, yt, xf, mean_y, actual) in enumerate(cache):
        model, optimizer = train_model(model, xt, yt, float(spec['tau']),
                                       float(spec['lambda']),
                                       EPOCHS_FIRST if i <= 1 else EPOCHS_NEXT,
                                       LR, optimizer=optimizer)
        if return_forecasts:
            with torch.no_grad():
                pred = float(model(xf).item()) + mean_y
            rows.append((date, actual, pred))
    if return_forecasts:
        return model, pd.DataFrame(rows, columns=['date', 'actual', 'prediction']).set_index('date')
    return model, None


def insample_variance(model, X, y, validation_cache):
    """Match fred.py insamp_pred: final model, fixed validation-end scaling."""
    dates = [item[0] for item in validation_cache]
    cutoff = dates[-1]
    # Original fred.py insamp_pred uses fixed end-of-validation scaling.
    x_hist = X.loc[:cutoff].iloc[:-1]
    y_hist = y.loc[:cutoff].iloc[:-1]
    x_mean, x_std = x_hist.mean(), x_hist.std(ddof=1)
    if x_std.isna().any() or (x_std == 0).any():
        raise ValueError('Undefined predictor standard deviation at validation end')
    x_eval = X.loc[dates].sub(x_mean).div(x_std)
    with torch.no_grad():
        predictions = model(torch.as_tensor(x_eval.to_numpy(), dtype=torch.float32)).numpy()
    predictions = np.asarray(predictions).reshape(-1) + float(y_hist.mean())
    return float(np.var(predictions, ddof=0))


def build_map(architectures, X, y, val_cache):
    """Fit all architecture/lambda candidates, then select nearest variance.

    The reference follows Domenico's test_script.py: first fitted variance
    after stable sorting candidates by lambda. Because multiple lambda=0
    candidates exist, the architecture ordering from the 04/05 search CSV is
    retained and the chosen reference is saved explicitly for audit.
    """
    grid = np.array([0.0] + [float(f'{v:.3f}') for v in
                     np.geomspace(0.02, 10.0, 40)], dtype=float)
    if len(np.unique(grid)) != len(grid):
        raise ValueError('Rounded regularization grid contains duplicates')
    all_mappings = {}
    for family in FAMILIES:
        mapped = []
        for tau in TAUS:
            group = architectures.loc[
                (architectures.family == family) & np.isclose(architectures.tau, tau)
            ]
            if group.empty:
                raise ValueError(f'No architectures for {family} q{tau:.2f}')
            saved = directory('insamp', family, f'q{tau:.2f}')
            values = []
            for arch in group.to_dict('records'):
                for j, lam in enumerate(grid):
                    spec = {k: arch[k] for k in
                            ('family', 'tau', 'nonlinear_layers', 'hidden_dim', 'alpha')}
                    spec['lambda'] = float(lam)
                    file = saved / f'{CACHE_VERSION}_{spec_id(spec)}.json'
                    if file.exists():
                        record = json.loads(file.read_text())
                        if any(record.get(k) != v for k, v in spec.items()):
                            raise ValueError(f'Cache specification mismatch: {file}')
                    else:
                        model, _ = fit_validation(spec, val_cache)
                        var = insample_variance(model, X, y, val_cache)
                        record = dict(**spec, variance=var, cache_version=CACHE_VERSION)
                        file.write_text(json.dumps(record))
                    if record.get('cache_version') != CACHE_VERSION:
                        raise ValueError(f'Cache version mismatch: {file}')
                    record = dict(record)
                    record['validation_loss'] = arch['best_validation_loss']
                    values.append(record)
                print(f'{family} q{tau:.2f}: fitted/cached {len(values)} '
                      f'of {len(group)*len(grid)} candidates', flush=True)
            candidates = pd.DataFrame(values).sort_values('lambda', kind='stable').reset_index(drop=True)
            if float(candidates.iloc[0]['lambda']) != 0.0:
                raise AssertionError('Expected lambda=0 reference')
            reference = float(candidates.iloc[0].variance)
            if not np.isfinite(reference) or reference <= 1e-12:
                raise ValueError(f'{family} q{tau:.2f}: invalid reference variance')
            candidates['achieved_r'] = candidates.variance / reference
            candidates['reference_variance'] = reference
            candidates.to_csv(directory('candidates') / f'{family}_q{tau:.2f}_insamp.csv', index=False)
            ref = candidates.iloc[0]
            print(f'REFERENCE {family} q{tau:.2f}: layers={int(ref.nonlinear_layers)}, '
                  f'dim={int(ref.hidden_dim)}, alpha={ref.alpha}, '
                  f'lambda={ref["lambda"]}, variance={reference:.8g}', flush=True)
            for r in R_GRID:
                ii = int(np.argmin(np.abs(candidates.variance.to_numpy() - r * reference)))
                choice = candidates.iloc[ii]
                record = choice.to_dict()
                record.update(target_r=float(r),
                              distance_to_target_r=abs(float(choice.achieved_r)-r))
                mapped.append(record)
                if record['distance_to_target_r'] > .06:
                    print(f'WARNING {family} q{tau:.2f} target={r:.1f} '
                          f'achieved={record["achieved_r"]:.3f}', flush=True)
        result = pd.DataFrame(mapped)
        result.to_csv(directory('mappings') / f'{family}.csv', index=False)
        all_mappings[family] = result
    return all_mappings


def naive(cache):
    return pd.DataFrame([(date, actual) for date, _, _, _, _, actual in cache],
                        columns=['date', 'actual']).set_index('date')


def benchmark(y, dates, tau):
    return np.array([np.quantile(y.loc[:d].iloc[:-1].to_numpy(), tau) for d in dates])


def hac_se(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return np.nan
    z = d - d.mean()
    lagmax = min(n-1, int(math.floor(4*(n/100)**(2/9))))
    lrv = z @ z / n
    for k in range(1, lagmax+1):
        lrv += 2*(1-k/(lagmax+1))*(z[k:] @ z[:-k])/n
    return math.sqrt(max(0, lrv)/n)


def test_forecast(model, spec, cache):
    optimizer = None  # Original fred.py restarts Adam at validation/test boundary.
    rows = []
    for date, xt, yt, xf, mean_y, actual in cache:
        model, optimizer = train_model(model, xt, yt, float(spec['tau']),
                                       float(spec['lambda']), EPOCHS_NEXT, LR,
                                       optimizer=optimizer)
        with torch.no_grad():
            pred = float(model(xf).item()) + mean_y
        rows.append((date, actual, pred))
    return pd.DataFrame(rows, columns=['date', 'actual', 'prediction']).set_index('date')


def create_tables(mappings, X, y, val_cache, test_cache):
    for family, mapping in mappings.items():
        forecast_folder = directory('forecasts', family)
        paths = {}
        unique = mapping.drop_duplicates(subset=['tau','nonlinear_layers','hidden_dim','alpha','lambda'])
        for i, row in enumerate(unique.to_dict('records')):
            spec = {k: row[k] for k in ('family', 'tau', 'nonlinear_layers', 'hidden_dim', 'alpha', 'lambda')}
            sid = spec_id(spec)
            vf, tf = forecast_folder / f'{sid}_validation.csv', forecast_folder / f'{sid}_test.csv'
            if not (vf.exists() and tf.exists()):
                print(f'{family} selected forecast {i+1}/{len(unique)}: {spec}', flush=True)
                model, val = fit_validation(spec, val_cache, return_forecasts=True)
                test = test_forecast(model, spec, test_cache)
                val.to_csv(vf)
                test.to_csv(tf)
            paths[sid] = (vf, tf)
        for sample, cache, which in [('validation', val_cache, 0), ('test', test_cache, 1)]:
            actual = naive(cache)
            dates = actual.index
            detailed, formatted = [], []
            naive_losses = {}
            for tau in TAUS:
                base = benchmark(y, dates, tau)
                naive_losses[tau] = pinball_loss_numpy(actual.actual.to_numpy(), base, tau)
            zero = dict(family=family, sample=sample, complexity=0.)
            for tau in TAUS:
                estimate = float(100 * np.mean(naive_losses[tau]) / (tau*(1-tau)))
                zero[f'{tau:.2f}'] = f'{estimate:.2f}'
                detailed.append(dict(family=family, sample=sample, target_r=0., tau=tau,
                                     achieved_r=0., estimate=estimate, hac_standard_error=np.nan,
                                     n_obs=len(dates)))
            formatted.append(zero)
            for r in R_GRID:
                line = dict(family=family, sample=sample, complexity=float(r))
                for tau in TAUS:
                    row = mapping[np.isclose(mapping.tau, tau) & np.isclose(mapping.target_r, r)].iloc[0]
                    spec = {k: row[k] for k in ('family','tau','nonlinear_layers','hidden_dim','alpha','lambda')}
                    f = pd.read_csv(paths[spec_id(spec)][which], index_col=0, parse_dates=True)
                    f = f.reindex(dates)
                    if f.prediction.isna().any():
                        raise ValueError(f'Forecast dates missing: {spec}, {sample}')
                    loss = pinball_loss_numpy(actual.actual.to_numpy(), f.prediction.to_numpy(), tau)
                    differential = 100*(loss-naive_losses[tau])/(tau*(1-tau))
                    estimate, se = float(np.mean(differential)), float(hac_se(differential))
                    line[f'{tau:.2f}'] = f'{estimate:.2f}\n({se:.2f})'
                    detailed.append(dict(family=family, sample=sample, target_r=float(r), tau=tau,
                                         achieved_r=float(row.achieved_r),
                                         distance_to_target_r=float(row.distance_to_target_r),
                                         estimate=estimate, hac_standard_error=se, n_obs=len(dates),
                                         nonlinear_layers=int(row.nonlinear_layers), hidden_dim=int(row.hidden_dim),
                                         alpha=float(row.alpha), **{'lambda': float(row['lambda'])}))
                formatted.append(line)
            stem = directory('tables') / f'table_{sample}_{family}'
            pd.DataFrame(detailed).to_csv(f'{stem}_detailed.csv', index=False)
            pd.DataFrame(formatted).to_csv(f'{stem}_formatted.csv', index=False)
            print(f'Saved {stem}_formatted.csv', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['map', 'tables', 'all'], default='all')
    args = parser.parse_args()
    X, y = load_replication_data()
    if X.isna().any().any() or y.isna().any():
        raise ValueError('Dataset contains NaNs; clean data before training')
    val_cache = prepare_cache(X, y, START_VAL, END_VAL)
    if args.stage in ('map', 'all'):
        architectures = search_architectures()
        mappings = build_map(architectures, X, y, val_cache)
    else:
        mappings = {f: pd.read_csv(ROOT / 'mappings' / f'{f}.csv') for f in FAMILIES}
    if args.stage in ('tables', 'all'):
        test_cache = prepare_cache(X, y, START_TEST, END_TEST)
        create_tables(mappings, X, y, val_cache, test_cache)


if __name__ == '__main__':
    main()
