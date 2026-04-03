import os
import shutil
import numpy as np
import pandas as pd

import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.expressions as ex
import biogeme.models as models
from biogeme.parameters import Parameters

from scipy import stats as sp_stats
from IPython.display import display


BETA_LABELS = {
    'ASC':                          r'ASC',
    'beta_z_n_pedestrians_acc':        r'\beta_{\text{Number of pedestrians}}',
    'beta_z_road_width_perp_m_acc':    r'\beta_{\text{Road width (m)}}',
    'beta_genre_female_acc':           r'\beta_{\text{Female (ref: Male)}}',
    'beta_z_hour_acc':                 r'\beta_{\text{Hour}}',
    'beta_z_age_acc':                  r'\beta_{\text{Age}}',
    "beta_experience_0_5_acc":         r'\beta_{\text{Experience < 6 months (ref: > 2 years)}}',
    "beta_experience_0_5_1_acc":       r'\beta_{\text{Experience 6 months--1 year (ref: > 2 years)}}',
    "beta_experience_1_2":         r'\beta_{\text{Experience 1--2 years (ref: > 2 years)}}',
    'beta_z_distance_km':          r'\beta_{\text{Total distance (km)}}',
    "SIGMA_ACC":                   r'\sigma_{\text{acc}}',
    "SIGMA_DEC":                   r'\sigma_{\text{dec}}'
}

def _df_to_latex_params(df, model_name):
    """Génère un tableau LaTeX des paramètres estimés."""

    col_value  = next(c for c in df.columns if 'value'   in c.lower())
    col_stderr = next(c for c in df.columns if 'std'     in c.lower())
    col_tstat  = next(c for c in df.columns if 't-stat'  in c.lower() or 'tstat' in c.lower())
    col_pvalue = next(c for c in df.columns if 'p-value' in c.lower() or 'pvalue' in c.lower())

    possible_name_cols = ['Name', 'name', 'Parameter', 'parameter', 'Beta', 'beta']
    col_param = next((c for c in possible_name_cols if c in df.columns), None)

    # Exclure les paramètres techniques (sigma, constantes internes)
    exclude_prefixes = ('log_sigma', 'mu0_', 'ls0_')

    lines = [
        r'\begin{table}[h!]\centering\small',
        r'\begin{tabular}{lrrrr}',
        r'\hline\hline',
        r'Paramètre & Valeur & Std. err. rob. & $t$-stat. rob. & $p$-valeur \\',
        r'\hline',
    ]

    unmatched = []

    for idx, row in df.iterrows():
        beta_name = str(row[col_param]).strip() if col_param else str(idx).strip()

        # Sauter les paramètres techniques
        if any(beta_name.startswith(p) for p in exclude_prefixes):
            continue

        if beta_name in BETA_LABELS:
            label = f'${BETA_LABELS[beta_name]}$'
        else:
            clean = (beta_name
                     .replace('beta_z_', '')
                     .replace('beta_', '')
                     .replace('_', ' '))
            label = rf'$\beta_{{\text{{{clean}}}}}$'
            unmatched.append(beta_name)

        sig = (r' $^{***}$' if row[col_pvalue] < .001 else
               r' $^{**}$'  if row[col_pvalue] < .01  else
               r' $^{*}$'   if row[col_pvalue] < .05  else '')

        lines.append(
            f'{label} & {row[col_value]:.4f} & {row[col_stderr]:.4f}'
            f' & {row[col_tstat]:.3f} & {row[col_pvalue]:.4f}{sig} \\\\'
        )

    lines += [
        r'\hline\hline',
        r'\end{tabular}',
        rf'\caption{{Estimated parameters of the linear regression model predicting the speed}}',
        rf'\label{{tab:{model_name}_params}}',
        r'\end{table}',
    ]

    if unmatched:
        print('\nBetas sans mapping explicite dans BETA_LABELS :')
        for b in sorted(set(unmatched)):
            print('  ', repr(b))

    return '\n'.join(lines)


def _metrics_to_latex(metrics, metrics_const, lrt_stat, lrt_df, lrt_p, model_name):
    """Génère un tableau LaTeX des statistiques générales du modèle logit."""

    if lrt_p is None or (isinstance(lrt_p, float) and np.isnan(lrt_p)):
        sig = ''
        lrt_p_txt = 'nan'
    else:
        sig = (
            r'$^{***}$' if lrt_p < .001 else
            r'$^{**}$'  if lrt_p < .01 else
            r'$^{*}$'   if lrt_p < .05 else
            ''
        )
        lrt_p_txt = f'{lrt_p:.4f}'

    rows = [
        (r'$N$',                            f'{metrics["N"]}'),
        (r'$K$',                            f'{metrics["K"]}'),
        (r'$\mathcal{LL}(\mathrm{null})$',  f'{metrics["LL_null"]:.2f}'),
        (r'$\mathcal{LL}(\mathrm{cst})$',   f'{metrics["LL_const"]:.2f}'),
        (r'$\mathcal{LL}(\hat{\beta})$',    f'{metrics["LL_final"]:.2f}'),
        (r'$\rho^2_{\mathrm{null}}$',       f'{metrics["rho2_null"]:.4f}'),
        (r'$\bar{\rho}^2_{\mathrm{null}}$', f'{metrics["rho2bar_null"]:.4f}'),
        (r'$\rho^2_{\mathrm{cst}}$',        f'{metrics["rho2_const"]:.4f}'),
        (r'$\bar{\rho}^2_{\mathrm{cst}}$',  f'{metrics["rho2bar_const"]:.4f}'),
        (r'AIC',                            f'{metrics["AIC"]:.2f}'),
        (r'BIC',                            f'{metrics["BIC"]:.2f}'),
        (rf'LRT $\chi^2({lrt_df})$ vs cst', f'{lrt_stat:.2f}{sig}'),
        (r'$p$-valeur LRT',                 lrt_p_txt),
    ]

    lines = [
        r'\begin{table}[h!]\centering\small',
        r'\begin{tabular}{lr}',
        r'\hline\hline',
        r'Statistique & Valeur \\',
        r'\hline',
    ] + [rf'{label} & {val} \\' for label, val in rows] + [
        r'\hline\hline',
        r'\end{tabular}',
        rf'\caption{{Statistics of the logit model predicting the speed change}}',
        rf'\label{{tab:{model_name}_stats}}',
        r'\end{table}',
    ]

    return '\n'.join(lines)


print('✔ Fonctions LaTeX chargées : _df_to_latex_params, _metrics_to_latex')

def _get_out_dir(model_name):
    base = os.path.join('model_results', model_name)
    if not os.path.exists(base):
        return base
    i = 2
    while os.path.exists(f'{base}_v{i}'):
        i += 1
    return f'{base}_v{i}'


def _extract_vars(expr):
    """Retourne les noms de colonnes-données (Variable) dans l'expression."""
    vars_ = []
    if isinstance(expr, ex.Variable):
        vars_.append(expr.name)
    for child in getattr(expr, 'children', []):
        vars_.extend(_extract_vars(child))
    return list(dict.fromkeys(vars_))


def _extract_betas(expr):
    """Retourne tous les nœuds Beta dans l'expression (récursif)."""
    betas = []
    if isinstance(expr, ex.Beta):
        betas.append(expr)
    for child in getattr(expr, 'children', []):
        betas.extend(_extract_betas(child))
    return betas


def _run_biogeme(database, logprob, model_name, params_bio, null_av=None):
    m = bio.BIOGEME(database, logprob, parameters=params_bio)
    m.model_name = model_name

    if null_av is None:
        # fallback binaire
        null_av = {0: ex.Numeric(1), 1: ex.Numeric(1)}

    m.calculate_null_loglikelihood(null_av)
    res = m.estimate()
    return m, res


def _compute_metrics(ll_f, ll_null, ll_const, n, nk, name):
    def safe_rho2(ll_ref, ll_f, nk=None):
        if ll_ref is None or ll_ref == 0:
            return np.nan
        if nk is None:
            return 1 - ll_f / ll_ref
        return 1 - (ll_f - nk) / ll_ref

    aic = -2 * ll_f + 2 * nk
    bic = -2 * ll_f + nk * np.log(n)

    return {
        "Model": name,
        "N": n,
        "K": nk,
        "LL_final": round(ll_f, 2),
        "LL_null": round(ll_null, 2) if ll_null is not None else np.nan,
        "LL_const": round(ll_const, 2) if ll_const is not None else np.nan,
        "rho2_null": round(safe_rho2(ll_null, ll_f), 4),
        "rho2bar_null": round(safe_rho2(ll_null, ll_f, nk), 4),
        "rho2_const": round(safe_rho2(ll_const, ll_f), 4),
        "rho2bar_const": round(safe_rho2(ll_const, ll_f, nk), 4),
        "AIC": round(aic, 2),
        "BIC": round(bic, 2),
    }


def run_mnl_3levels(
    df_est,
    utility_decelerate,
    utility_accelerate,
    model_name,
    choice_col='choice_3'
):
    """
    Multinomial logit à 3 alternatives :
        0 = décélérer
        1 = maintenir
        2 = accélérer

    Normalisation :
        U_maintain = 0
    """

    # ── 1. Données ─────────────────────────────────────────────────────────
    vars_dec = _extract_vars(utility_decelerate)
    vars_acc = _extract_vars(utility_accelerate)

    cols = [choice_col] + list(dict.fromkeys(vars_dec + vars_acc))

    before = len(df_est)
    data = df_est[cols].dropna().copy()
    dropped = before - len(data)

    if dropped:
        print(f'[{model_name}] ⚠ {dropped} lignes supprimées (NaN)')

    for col in data.columns:
        if pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype(int)

    # garder uniquement les 3 modalités valides
    data = data[data[choice_col].isin([0, 1, 2])].copy()

    N = len(data)
    database = db.Database(model_name, data)

    # ── 2. Paramètres Biogeme ──────────────────────────────────────────────
    params_bio = Parameters()
    params_bio.set_value('generate_html', True)
    params_bio.set_value('generate_yaml', False)
    params_bio.set_value('generate_netcdf', False)
    params_bio.set_value('number_of_threads', 8)

    choice = ex.Variable(choice_col)

    # Normalisation
    U_maintain = ex.Numeric(0)

    # ── 3. Modèle constant seul ────────────────────────────────────────────
    const_name = f'{model_name}_constant'

    ASC_DEC = ex.Beta('ASC_DEC', 0, None, None, 0)
    ASC_ACC = ex.Beta('ASC_ACC', 0, None, None, 0)
    # U_maintain = 0 => pas d'ASC pour maintain

    V_const = {
        0: ASC_DEC,
        1: U_maintain,
        2: ASC_ACC
    }
    av = {0: 1, 1: 1, 2: 1}

    logprob_const = models.loglogit(V_const, av, choice)
    db_const = db.Database(const_name, data)
    _, res_const = _run_biogeme(
        db_const,
        logprob_const,
        const_name,
        params_bio,
        null_av={0: ex.Numeric(1), 1: ex.Numeric(1), 2: ex.Numeric(1)}
    )

    ll_const = res_const.raw_estimation_results.final_log_likelihood
    ll_null = res_const.raw_estimation_results.null_log_likelihood
    k_const = len(res_const.get_beta_values())

    # ── 4. Modèle principal ────────────────────────────────────────────────
    V = {
        0: utility_decelerate,
        1: U_maintain,
        2: utility_accelerate
    }

    logprob = models.loglogit(V, av, choice)

    _, res = _run_biogeme(
        database,
        logprob,
        model_name,
        params_bio,
        null_av={0: ex.Numeric(1), 1: ex.Numeric(1), 2: ex.Numeric(1)}
    )

    # ── 5. Métriques ───────────────────────────────────────────────────────
    final_ll = res.raw_estimation_results.final_log_likelihood
    k = len(res.get_beta_values())

    metrics_const = _compute_metrics(
        ll_f=ll_const,
        ll_null=ll_null,
        ll_const=ll_const,
        n=N,
        nk=k_const,
        name=const_name,
    )

    metrics = _compute_metrics(
        ll_f=final_ll,
        ll_null=ll_null,
        ll_const=ll_const,
        n=N,
        nk=k,
        name=model_name,
    )

    lrt_stat = -2 * (ll_const - final_ll)
    lrt_df = k - k_const
    lrt_p = sp_stats.chi2.sf(lrt_stat, lrt_df) if lrt_df > 0 else np.nan

    metrics['LRT_stat'] = round(lrt_stat, 2)
    metrics['LRT_df'] = lrt_df
    metrics['LRT_p'] = round(lrt_p, 4) if not np.isnan(lrt_p) else np.nan

    # ── 6. params_df ───────────────────────────────────────────────────────
    params_df = res.get_estimated_parameters()
    params_df.columns = [c.strip() for c in params_df.columns]

    # ── 7. Sauvegarde ──────────────────────────────────────────────────────
    out_dir = _get_out_dir(model_name)
    os.makedirs(out_dir, exist_ok=True)

    if '_df_to_latex_params' in globals():
        with open(os.path.join(out_dir, f'{model_name}_params.tex'), 'w') as f:
            f.write(_df_to_latex_params(params_df, model_name))

    if '_metrics_to_latex' in globals():
        with open(os.path.join(out_dir, f'{model_name}_stats.tex'), 'w') as f:
            f.write(
                _metrics_to_latex(
                    metrics,
                    metrics_const,
                    lrt_stat,
                    lrt_df,
                    lrt_p,
                    model_name
                )
            )

    for name in [model_name, const_name]:
        for prefix in ['']:
            for ext in ['.iter', '.html']:
                fname = f'{prefix}{name}{ext}'
                if os.path.exists(fname):
                    shutil.move(fname, os.path.join(out_dir, fname))

    # ── 8. Affichage ───────────────────────────────────────────────────────
    print(f'\n{"="*90}')
    print(f'  Modèle constant  ({const_name})')
    print(
        f'  LL={ll_const:.2f}  '
        f'rho2_null={metrics_const["rho2_null"]:.4f}  '
        f'rho2bar_null={metrics_const["rho2bar_null"]:.4f}  '
        f'rho2_const={metrics_const["rho2_const"]:.4f}  '
        f'rho2bar_const={metrics_const["rho2bar_const"]:.4f}  '
        f'AIC={metrics_const["AIC"]:.1f}  '
        f'BIC={metrics_const["BIC"]:.1f}'
    )
    print(f'{"─"*90}')
    print(f'  Modèle principal ({model_name})')
    print(
        f'  N={N}  K={k}  LL={final_ll:.2f}  '
        f'rho2_null={metrics["rho2_null"]:.4f}  '
        f'rho2bar_null={metrics["rho2bar_null"]:.4f}  '
        f'rho2_const={metrics["rho2_const"]:.4f}  '
        f'rho2bar_const={metrics["rho2bar_const"]:.4f}  '
        f'AIC={metrics["AIC"]:.1f}  '
        f'BIC={metrics["BIC"]:.1f}'
    )
    print(f'{"─"*90}')

    if np.isnan(lrt_p):
        sig = ''
        p_txt = 'nan'
    else:
        sig = (
            ' ***' if lrt_p < 0.001 else
            ' **' if lrt_p < 0.01 else
            ' *' if lrt_p < 0.05 else
            ' (n.s.)'
        )
        p_txt = f'{lrt_p:.4f}'

    print(f'  LRT vs constant : χ²({lrt_df})={lrt_stat:.2f}  p={p_txt}{sig}')
    print(f'{"="*90}')

    try:
        display(params_df.style.format({
            'Value': '{:.4f}',
            'Std err.': '{:.4f}',
            'Robust std err.': '{:.4f}',
            't-stat.': '{:.3f}',
            'Robust t-stat.': '{:.3f}',
            'p-value': '{:.4f}',
            'Robust p-value': '{:.4f}',
        }))
    except:
        display(params_df)

    return res, params_df, metrics, metrics_const
def run_mxl_panel_3levels(
    df_est,
    utility_decelerate,
    utility_accelerate,
    model_name,
    panel_id_col='rider_id_num',
    choice_col='choice_3',
    number_of_draws=1,
    draw_type='HALTON',
    random_number_generators=None,
    calculate_null=True,
):
    """
    Mixed Logit panel à 3 alternatives :
        0 = décélérer
        1 = maintenir
        2 = accélérer

    Normalisation :
        U_maintain = 0
    """

    import os
    import shutil
    import time
    import numpy as np
    import pandas as pd
    from IPython.display import display

    print("\n" + "=" * 100)
    print(f"🚀 LANCEMENT ESTIMATION : {model_name}")
    print("=" * 100)

    # ── 1. Colonnes nécessaires ────────────────────────────────────────────
    print(f"[{model_name}] Étape 1/6 - Extraction des variables des utilités")

    vars_dec = _extract_vars(utility_decelerate)
    vars_acc = _extract_vars(utility_accelerate)

    cols = [panel_id_col, choice_col] + list(dict.fromkeys(vars_dec + vars_acc))
    print(f"[{model_name}] Variables utilité décélérer : {vars_dec}")
    print(f"[{model_name}] Variables utilité accélérer : {vars_acc}")
    print(f"[{model_name}] Colonnes retenues : {cols}")

    before = len(df_est)
    print(f"[{model_name}] Nombre de lignes avant nettoyage : {before:,}")

    data = df_est[cols].dropna().copy()
    dropped = before - len(data)

    if dropped:
        print(f'[{model_name}] ⚠ {dropped:,} lignes supprimées (NaN)')
    else:
        print(f'[{model_name}] Aucune ligne supprimée pour NaN')

    print(f"[{model_name}] Conversion des booléens en entiers si nécessaire")
    for col in data.columns:
        if pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype(int)
            print(f"[{model_name}]   - Colonne convertie bool -> int : {col}")

    before_choice = len(data)
    data = data[data[choice_col].isin([0, 1, 2])].copy()
    removed_choice = before_choice - len(data)

    if removed_choice:
        print(f'[{model_name}] ⚠ {removed_choice:,} lignes supprimées (choix invalides)')
    else:
        print(f'[{model_name}] Aucun choix invalide supprimé')

    if data.empty:
        raise ValueError(f"[{model_name}] Aucune observation valide après nettoyage.")

    print(f"[{model_name}] Tri des données par identifiant panel : {panel_id_col}")
    data = data.sort_values(panel_id_col).reset_index(drop=True)

    N_obs = len(data)
    N_ind = data[panel_id_col].nunique()

    print(f"[{model_name}] Données prêtes")
    print(f"[{model_name}]   - Observations : {N_obs:,}")
    print(f"[{model_name}]   - Individus    : {N_ind:,}")

    # ── 2. Database panel ──────────────────────────────────────────────────
    print(f"[{model_name}] Étape 2/6 - Création de la base panel Biogeme")

    database = db.Database(model_name, data)
    database.panel(panel_id_col)

    choice = ex.Variable(choice_col)
    U_maintain = ex.Numeric(0)

    av = {0: 1, 1: 1, 2: 1}

    print(f"[{model_name}] Database panel créée avec succès")

    # ── 3. Modèle mixed logit panel ────────────────────────────────────────
    print(f"[{model_name}] Étape 3/6 - Construction du modèle mixed logit panel")

    V = {
        0: utility_decelerate,
        1: U_maintain,
        2: utility_accelerate,
    }

    p_obs = models.logit(V, av, choice)
    p_traj = ex.PanelLikelihoodTrajectory(p_obs)
    logprob = ex.log(ex.MonteCarlo(p_traj))

    params_bio = Parameters()
    params_bio.set_value('generate_html', True)
    params_bio.set_value('generate_yaml', False)
    params_bio.set_value('generate_netcdf', False)
    params_bio.set_value('number_of_draws', number_of_draws)

    print(f"[{model_name}] Paramètres Biogeme du modèle principal")
    print(f"[{model_name}]   - draw_type        : {draw_type}")
    print(f"[{model_name}]   - number_of_draws  : {number_of_draws}")
    print(f"[{model_name}]   - RNG custom       : {'oui' if random_number_generators is not None else 'non'}")

    m = bio.BIOGEME(
        database,
        logprob,
        parameters=params_bio,
        random_number_generators=random_number_generators,
        number_of_draws=number_of_draws,
        calculating_second_derivatives='never',
    )
    m.model_name = model_name

    print(f"[{model_name}] Vérification de l'objet BIOGEME principal")
    try:
        print(f"[{model_name}]   - number_of_draws effectif dans m : {m.number_of_draws}")
    except Exception:
        print(f"[{model_name}]   - impossible de lire m.number_of_draws")

    if calculate_null:
        print(f"[{model_name}] Calcul de la null log-likelihood")
        m.calculate_null_loglikelihood({
            0: ex.Numeric(1),
            1: ex.Numeric(1),
            2: ex.Numeric(1),
        })

    print(f"[{model_name}] Estimation du modèle principal : {model_name}")
    print(f"[{model_name}] Début Monte-Carlo / maximum likelihood...")
    t0_main = time.time()
    res = m.estimate()
    t1_main = time.time()

    print(f"[{model_name}] ✅ Modèle principal estimé en {t1_main - t0_main:.2f} sec")

    # ── 4. Métriques ───────────────────────────────────────────────────────
    print(f"[{model_name}] Étape 4/6 - Calcul des métriques")

    final_ll = res.raw_estimation_results.final_log_likelihood
    ll_null = (
        res.raw_estimation_results.null_log_likelihood
        if calculate_null else np.nan
    )
    k = len(res.get_beta_values())

    metrics = _compute_metrics(
        ll_f=final_ll,
        ll_null=ll_null,
        ll_const=np.nan,
        n=N_ind,
        nk=k,
        name=model_name,
    )

    metrics['N_obs'] = N_obs
    metrics['N_individuals'] = N_ind
    metrics['draw_type'] = draw_type
    metrics['number_of_draws'] = number_of_draws

    print(f"[{model_name}] Métriques calculées")
    print(f"[{model_name}]   - Final LL : {final_ll:.6f}")
    print(f"[{model_name}]   - LL null  : {ll_null if np.isnan(ll_null) else round(ll_null, 6)}")
    print(f"[{model_name}]   - K        : {k}")

    # ── 5. Paramètres estimés ──────────────────────────────────────────────
    print(f"[{model_name}] Étape 5/6 - Extraction des paramètres estimés")

    params_df = res.get_estimated_parameters()
    params_df.columns = [c.strip() for c in params_df.columns]

    print(f"[{model_name}] Paramètres extraits : {len(params_df)} lignes")

    # ── 6. Sauvegarde + affichage ──────────────────────────────────────────
    print(f"[{model_name}] Étape 6/6 - Sauvegarde des sorties")

    out_dir = _get_out_dir(model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{model_name}] Dossier de sortie : {out_dir}")

    if '_df_to_latex_params' in globals():
        tex_params_path = os.path.join(out_dir, f'{model_name}_params.tex')
        with open(tex_params_path, 'w') as f:
            f.write(_df_to_latex_params(params_df, model_name))
        print(f"[{model_name}] Fichier paramètres LaTeX sauvegardé : {tex_params_path}")

    if '_metrics_to_latex' in globals():
        tex_stats_path = os.path.join(out_dir, f'{model_name}_stats.tex')
        with open(tex_stats_path, 'w') as f:
            f.write(
                _metrics_to_latex(
                    metrics,
                    None,
                    np.nan,
                    np.nan,
                    np.nan,
                    model_name
                )
            )
        print(f"[{model_name}] Fichier métriques LaTeX sauvegardé : {tex_stats_path}")

    for prefix in ['']:
        for ext in ['.iter', '.html']:
            fname = f'{prefix}{model_name}{ext}'
            if os.path.exists(fname):
                src = fname
                dst = os.path.join(out_dir, fname)
                shutil.move(src, dst)
                print(f"[{model_name}] Fichier déplacé : {src} -> {dst}")

    print(f'\n{"="*100}')
    print(f'  Mixed logit panel ({model_name})')
    print(f'  Individus={N_ind:,}  Observations={N_obs:,}  Draws={number_of_draws:,} ({draw_type})')
    print(f'{"─"*100}')
    print(
        f'  K={k}  LL={final_ll:.2f}  '
        f'rho2_null={metrics["rho2_null"]:.4f}  '
        f'rho2bar_null={metrics["rho2bar_null"]:.4f}  '
        f'AIC={metrics["AIC"]:.1f}  '
        f'BIC={metrics["BIC"]:.1f}'
    )
    print(f'{"="*100}')

    try:
        display(params_df.style.format({
            'Value': '{:.4f}',
            'Std err.': '{:.4f}',
            'Robust std err.': '{:.4f}',
            't-stat.': '{:.3f}',
            'Robust t-stat.': '{:.3f}',
            'p-value': '{:.4f}',
            'Robust p-value': '{:.4f}',
        }))
    except Exception:
        display(params_df)

    print(f"[{model_name}] ✅ FIN de run_mxl_panel_3levels")
    print("=" * 100)

    return res, params_df, metrics