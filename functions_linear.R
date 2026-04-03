# ══════════════════════════════════════════════════════════════════════════════
# functions_linear.R
# Equivalent R de funtion_linear.py
# Packages requis : lme4, lmerTest
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)   # ajoute les p-valeurs aux lmer (approx. Satterthwaite)
})

# ── Labels LaTeX ──────────────────────────────────────────────────────────────
BETA_LABELS <- c(
  "(Intercept)"           = "\\mu",
  "z_n_pedestrians"       = "\\beta_{\\text{Number of pedestrians}}",
  "z_road_width_perp_m"   = "\\beta_{\\text{Road width (m)}}",
  "genre_femaleTRUE"      = "\\beta_{\\text{Female (ref: Male)}}",
  "genre_female"          = "\\beta_{\\text{Female (ref: Male)}}",
  "z_hour"                = "\\beta_{\\text{Hour}}",
  "z_age"                 = "\\beta_{\\text{Age}}",
  "experience_0.5-1"      = "\\beta_{\\text{Experience 6 months--1 year (ref: > 2 years)}}",
  "experience_1-2"        = "\\beta_{\\text{Experience 1--2 years (ref: > 2 years)}}",
  "experience_<0.5"       = "\\beta_{\\text{Experience < 6 months (ref: > 2 years)}}",
  "z_distance_km"         = "\\beta_{\\text{Total distance (km)}}"
)

# ── Utilitaires internes ──────────────────────────────────────────────────────
.get_out_dir <- function(model_name) {
  base <- file.path("model_results_linear", model_name)
  if (!dir.exists(base)) return(base)
  i <- 2
  while (dir.exists(paste0(base, "_v", i))) i <- i + 1
  paste0(base, "_v", i)
}

.sig_stars <- function(p) {
  ifelse(is.na(p), "",
  ifelse(p < 0.001, " $^{***}$",
  ifelse(p < 0.01,  " $^{**}$",
  ifelse(p < 0.05,  " $^{*}$", ""))))
}

.beta_label <- function(name) {
  if (name %in% names(BETA_LABELS)) {
    paste0("$", BETA_LABELS[name], "$")
  } else {
    clean <- gsub("TRUE$", "", name)
    clean <- gsub("z_", "", clean)
    clean <- gsub("_", " ", clean)
    paste0("$\\beta_{\\text{", clean, "}}$")
  }
}

# ── Export LaTeX : tableau des paramètres ─────────────────────────────────────
.params_to_latex <- function(params_df, model_name) {
  skip_patterns <- c("^sd_", "^cor_", "^sigma$", "^Residual$")

  lines <- c(
    "\\begin{table}[h!]\\centering\\small",
    "\\begin{tabular}{lrrrr}",
    "\\hline\\hline",
    "Param\\`etre & Valeur & Std. err. & $t$-stat. & $p$-valeur \\\\",
    "\\hline"
  )

  for (nm in rownames(params_df)) {
    if (any(sapply(skip_patterns, function(p) grepl(p, nm)))) next

    est <- params_df[nm, "Estimate"]
    se  <- params_df[nm, "Std. Error"]
    tv  <- params_df[nm, "t value"]
    pv  <- if ("Pr(>|t|)" %in% colnames(params_df)) params_df[nm, "Pr(>|t|)"] else NA

    sig  <- .sig_stars(pv)
    pstr <- if (is.na(pv)) "---" else sprintf("%.4f", pv)
    lines <- c(lines,
      sprintf("%s & %.4f & %.4f & %.3f & %s%s \\\\",
              .beta_label(nm), est, se, tv, pstr, sig))
  }

  lines <- c(lines,
    "\\hline\\hline",
    "\\end{tabular}",
    paste0("\\caption{Estimated parameters of the linear regression",
           " model predicting the speed}"),
    paste0("\\label{tab:", model_name, "_params}"),
    "\\end{table}"
  )
  paste(lines, collapse = "\n")
}

# ── Export LaTeX : tableau des statistiques du modèle ─────────────────────────
.stats_to_latex <- function(metrics, model_name) {
  lrt_p <- metrics$LRT_p
  sig   <- if (is.na(lrt_p)) "" else .sig_stars(lrt_p)

  rows <- list(
    c("$N$",                                sprintf("%d",   metrics$N)),
    c("$K$",                                sprintf("%d",   metrics$K)),
    c("$\\mathcal{L}(\\text{nul})$",        sprintf("%.2f", metrics$LL_null)),
    c("$\\mathcal{L}(\\hat{\\beta})$",      sprintf("%.2f", metrics$LL_final)),
    c("$\\bar{\\rho}^2$",                   sprintf("%.4f", metrics$rho2_bar)),
    c(sprintf("LRT $\\chi^2(%d)$ vs nul",   metrics$LRT_df),
      sprintf("%.2f%s", metrics$LRT_stat,   sig)),
    c("$p$-valeur LRT",
      if (is.na(lrt_p)) "---" else sprintf("%.4f", lrt_p))
  )

  # Lignes effets aléatoires (modèle mixte uniquement)
  if (!is.null(metrics$sigma_rider)) {
    sigma_rid_p <- if (!is.null(metrics$sigma_rider_p)) metrics$sigma_rider_p else NA
    sig_re <- .sig_stars(sigma_rid_p)
    p_str  <- if (is.na(sigma_rid_p)) "---" else sprintf("%.2e", sigma_rid_p)
    rows <- c(rows, list(
      c("$\\sigma_{\\text{rider}}$",
        sprintf("%.4f km/h", metrics$sigma_rider)),
      c("$\\sigma_{\\varepsilon}$",
        sprintf("%.4f km/h", metrics$sigma_eps)),
      c("ICC",
        sprintf("%.4f", metrics$ICC)),
      c(sprintf("LRT $\\sigma_{\\text{rider}}=0$  $\\chi^2(1)$"),
        sprintf("%s%s", p_str, sig_re))
    ))
  }

  lines <- c(
    "\\begin{table}[h!]\\centering\\small",
    "\\begin{tabular}{lr}",
    "\\hline\\hline",
    "Statistique & Valeur \\\\",
    "\\hline"
  )
  for (r in rows) lines <- c(lines, paste0(r[[1]], " & ", r[[2]], " \\\\"))
  lines <- c(lines,
    "\\hline\\hline",
    "\\end{tabular}",
    paste0("\\caption{Statistics of the linear regression",
           " model predicting the speed}"),
    paste0("\\label{tab:", model_name, "_stats}"),
    "\\end{table}"
  )
  paste(lines, collapse = "\n")
}

# ══════════════════════════════════════════════════════════════════════════════
# run_linear(df_est, rhs, model_name)
#
# Estime un modèle linéaire gaussien (OLS = MLE gaussien).
#   y = rhs + epsilon,  epsilon ~ N(0, sigma²)
#
# Arguments
#   df_est     : data.frame avec au moins speed_kmh_t1 + les variables de rhs
#   rhs        : partie droite de la formule, ex. "z_n_pedestrians + genre_female"
#                Utilisez "1" pour le modèle nul (intercept seul).
#   model_name : identifiant du modèle (ex. "M1_pedestrians")
#
# Retour : liste(fit, params, metrics)
# ══════════════════════════════════════════════════════════════════════════════
run_linear <- function(df_est, rhs, model_name) {

  formula_obj  <- as.formula(paste("speed_kmh_t1 ~", rhs))
  formula_null <- speed_kmh_t1 ~ 1

  vars_used <- unique(c("speed_kmh_t1", all.vars(formula_obj)))
  vars_used <- vars_used[vars_used %in% names(df_est)]
  data      <- df_est[, vars_used, drop = FALSE]
  before    <- nrow(data)
  data      <- data[complete.cases(data), ]
  dropped   <- before - nrow(data)
  if (dropped > 0) message(sprintf("[%s] ⚠ %d lignes supprimées (NaN)", model_name, dropped))
  N <- nrow(data)

  # ── Estimation ────────────────────────────────────────────────────────────
  fit  <- lm(formula_obj,  data = data)
  fit0 <- lm(formula_null, data = data)

  ll      <- as.numeric(logLik(fit))
  ll_null <- as.numeric(logLik(fit0))
  k       <- length(coef(fit)) + 1   # coefs + sigma
  k_null  <- 2                        # mu + sigma

  # ── Métriques ─────────────────────────────────────────────────────────────
  rho2     <- 1 - ll / ll_null
  rho2_bar <- 1 - (ll - k) / ll_null
  aic_val  <- AIC(fit)
  bic_val  <- BIC(fit)
  lrt_stat <- -2 * (ll_null - ll)
  lrt_df   <- k - k_null
  lrt_p    <- if (lrt_df > 0) pchisq(lrt_stat, df = lrt_df, lower.tail = FALSE) else NA

  sig_str <- if (!is.na(lrt_p) && lrt_p < 0.001) " ***" else
             if (!is.na(lrt_p) && lrt_p < 0.01)  " **"  else
             if (!is.na(lrt_p) && lrt_p < 0.05)  " *"   else " (n.s.)"

  cat(sprintf("\n%s\n", strrep("=", 65)))
  cat(sprintf("  Modèle nul      : LL=%.2f\n", ll_null))
  cat(sprintf("  Modèle principal: N=%d  K=%d  LL=%.2f\n", N, k, ll))
  cat(sprintf("  rho²=%.4f  AIC=%.1f  BIC=%.1f\n", rho2, aic_val, bic_val))
  cat(sprintf("  LRT vs nul : chi²(%d)=%.2f  p=%.4f%s\n",
              lrt_df, lrt_stat, lrt_p, sig_str))
  cat(sprintf("%s\n", strrep("=", 65)))
  print(summary(fit)$coefficients)

  metrics <- list(
    Model    = model_name,
    N        = N,
    K        = k,
    LL_null  = round(ll_null,  2),
    LL_final = round(ll,       2),
    rho2     = round(rho2,     4),
    rho2_bar = round(rho2_bar, 4),
    AIC      = round(aic_val,  2),
    BIC      = round(bic_val,  2),
    LRT_stat = round(lrt_stat, 2),
    LRT_df   = lrt_df,
    LRT_p    = if (!is.na(lrt_p)) round(lrt_p, 4) else NA
  )

  # ── Sauvegarde ────────────────────────────────────────────────────────────
  out_dir   <- .get_out_dir(model_name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  params_df <- as.data.frame(summary(fit)$coefficients)
  write.csv(params_df, file.path(out_dir, paste0(model_name, "_params.csv")))
  writeLines(.params_to_latex(params_df, model_name),
             file.path(out_dir, paste0(model_name, "_params.tex")))
  writeLines(.stats_to_latex(metrics, model_name),
             file.path(out_dir, paste0(model_name, "_stats.tex")))

  invisible(list(fit = fit, params = params_df, metrics = metrics))
}

# ══════════════════════════════════════════════════════════════════════════════
# run_mixed_linear_panel(df_est, rhs, model_name, panel_id_col)
#
# Estime un modèle linéaire mixte gaussien en panel.
#   y_it = rhs_it + u_i + e_it
#   u_i  ~ N(0, sigma_rider²)   (intercept aléatoire par rider)
#   e_it ~ N(0, sigma_eps²)
#
# Arguments
#   df_est       : data.frame
#   rhs          : partie fixe, ex. "z_n_pedestrians + genre_female"
#   model_name   : identifiant du modèle
#   panel_id_col : colonne identifiant l'individu (défaut : "rider_id")
#
# Retour : liste(fit, params, metrics)
# ══════════════════════════════════════════════════════════════════════════════
run_mixed_linear_panel <- function(df_est, rhs, model_name,
                                   panel_id_col = "rider_id") {

  formula_obj  <- as.formula(
    paste0("speed_kmh_t1 ~ ", rhs, " + (1|", panel_id_col, ")")
  )
  formula_null <- as.formula(
    paste0("speed_kmh_t1 ~ (1|", panel_id_col, ")")
  )

  vars_used <- unique(c("speed_kmh_t1", panel_id_col,
                         all.vars(as.formula(paste("~", rhs)))))
  vars_used <- vars_used[vars_used %in% names(df_est)]
  data      <- df_est[, vars_used, drop = FALSE]
  before    <- nrow(data)
  data      <- data[complete.cases(data), ]
  dropped   <- before - nrow(data)
  if (dropped > 0) message(sprintf("[%s] ⚠ %d lignes supprimées (NaN)", model_name, dropped))

  # Trier par individu (bonne pratique panel)
  data <- data[order(data[[panel_id_col]]), ]

  N_obs    <- nrow(data)
  N_riders <- length(unique(data[[panel_id_col]]))

  # ── Estimation ML (REML=FALSE obligatoire pour LRT valide) ────────────────
  fit  <- lmer(formula_obj,  data = data, REML = FALSE)
  fit0 <- lmer(formula_null, data = data, REML = FALSE)

  ll      <- as.numeric(logLik(fit))
  ll_null <- as.numeric(logLik(fit0))
  k       <- attr(logLik(fit),  "df")
  k_null  <- attr(logLik(fit0), "df")

  # ── Métriques ─────────────────────────────────────────────────────────────
  rho2     <- 1 - ll / ll_null
  rho2_bar <- 1 - (ll - k) / ll_null
  aic_val  <- AIC(fit)
  bic_val  <- BIC(fit)
  lrt_stat <- -2 * (ll_null - ll)
  lrt_df   <- k - k_null
  lrt_p    <- if (lrt_df > 0) pchisq(lrt_stat, df = lrt_df, lower.tail = FALSE) else NA

  sig_str <- if (!is.na(lrt_p) && lrt_p < 0.001) " ***" else
             if (!is.na(lrt_p) && lrt_p < 0.01)  " **"  else
             if (!is.na(lrt_p) && lrt_p < 0.05)  " *"   else " (n.s.)"

  # Effets aléatoires
  re_df     <- as.data.frame(VarCorr(fit))
  sigma_rid <- sqrt(re_df$vcov[re_df$grp == panel_id_col])
  sigma_eps <- sqrt(re_df$vcov[re_df$grp == "Residual"])

  cat(sprintf("\n%s\n", strrep("=", 72)))
  cat(sprintf("  Mixed panel linear model: %s\n", model_name))
  cat(sprintf("  Riders=%d  Observations=%d\n", N_riders, N_obs))
  cat(sprintf("  sigma_rider=%.4f  sigma_eps=%.4f\n", sigma_rid, sigma_eps))
  cat(sprintf("  Modèle nul      : LL=%.2f\n", ll_null))
  cat(sprintf("  Modèle principal: K=%d  LL=%.2f\n", k, ll))
  cat(sprintf("  rho²=%.4f  AIC=%.1f  BIC=%.1f\n", rho2, aic_val, bic_val))
  cat(sprintf("  LRT vs nul : chi²(%d)=%.2f  p=%.4f%s\n",
              lrt_df, lrt_stat, lrt_p, sig_str))
  cat(sprintf("%s\n", strrep("=", 72)))
  print(summary(fit)$coefficients)

  icc <- sigma_rid^2 / (sigma_rid^2 + sigma_eps^2)

  # p-value pour sigma_rider via ranova (LRT sigma=0)
  ranova_res    <- ranova(fit)
  sigma_rider_p <- ranova_res[["Pr(>Chisq)"]][2]   # ligne (1|panel_id_col)

  metrics <- list(
    Model         = model_name,
    N             = N_obs,
    N_riders      = N_riders,
    K             = k,
    LL_null       = round(ll_null,      2),
    LL_final      = round(ll,           2),
    rho2          = round(rho2,         4),
    rho2_bar      = round(rho2_bar,     4),
    AIC           = round(aic_val,      2),
    BIC           = round(bic_val,      2),
    LRT_stat      = round(lrt_stat,     2),
    LRT_df        = lrt_df,
    LRT_p         = if (!is.na(lrt_p)) round(lrt_p, 4) else NA,
    sigma_rider   = round(sigma_rid,    4),
    sigma_eps     = round(sigma_eps,    4),
    ICC           = round(icc,          4),
    sigma_rider_p = sigma_rider_p
  )

  # ── Sauvegarde ────────────────────────────────────────────────────────────
  out_dir   <- .get_out_dir(model_name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  params_df <- as.data.frame(summary(fit)$coefficients)
  write.csv(params_df, file.path(out_dir, paste0(model_name, "_params.csv")))
  writeLines(.params_to_latex(params_df, model_name),
             file.path(out_dir, paste0(model_name, "_params.tex")))
  writeLines(.stats_to_latex(metrics, model_name),
             file.path(out_dir, paste0(model_name, "_stats.tex")))

  invisible(list(fit = fit, params = params_df, metrics = metrics))
}

cat("✔ Fonctions R chargées : run_linear, run_mixed_linear_panel\n")
