# ══════════════════════════════════════════════════════════════════════════════
# functions_linear.R
# Equivalent R de funtion_linear.py
# Packages requis : lme4, lmerTest
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)   # ajoute les p-valeurs aux lmer (approx. Satterthwaite)
  if (!requireNamespace("MuMIn", quietly = TRUE)) install.packages("MuMIn", repos = "https://cloud.r-project.org")
  library(MuMIn)      # r.squaredGLMM : R² marginal et conditionnel
})

# ── Labels LaTeX ──────────────────────────────────────────────────────────────
# Seule table de labels : variable R → texte d'affichage (sans LaTeX).
# Utilisée à la fois pour les β et pour les symboles de variables (x̃, 𝟙, x).
ID_LABELS <- c(
  "z_n_pedestrians"            = "N visible pedestrians < 15m",
  "z_road_width_perp_m"        = "Road width (m)",
  "z_n_elderly"                = "N elderly",
  "z_n_children"               = "N children",
  "z_n_running"                = "N running pedestrians",
  "z_hour"                     = "Hour",
  "z_age"                      = "Age",
  "distance_km"                = "Distance (km)",
  "genrefemale"                = "Female (ref : Male)",
  "genremale"                  = "Male (ref : Female)",
  "at_intersection1"           = "Intersection (ref : No)",
  "SURFACE_CONDITION_LABELWet" = "Wet surface",
  "SURFACE_CONDITION_LABELDry" = "Dry surface",
  "time_of_dayAfternoon"       = "Afternoon (ref : Morning)",
  "time_of_dayNight"           = "Evening/Night",
  "experience0.5-1"            = "Exp. 6mo--1yr",
  "experience1-2"              = "Exp. 1--2yr",
  "experience<0.5"             = "Exp. <6mo",
  "prop_vru_cyclist"            = "Prop. cyclist",
  "prop_vru_pedestrian"         = "Proportion of pedestrians <15m among VRUs",
  "prop_interaction_same_direction" = "Proportion of same-direction encounters",
  "prop_interaction_opposite_direction" = "Proportion of opposite-direction encounters",
  "prop_interaction_crossing" = "Proportion of crossing encounters"
)

# Texte d'affichage pour un nom R — lookup dans ID_LABELS, sinon auto-génération.
.id_text <- function(nm) {
  if (nm %in% names(ID_LABELS)) return(ID_LABELS[[nm]])
  clean <- sub("TRUE$", "", nm)
  clean <- sub("^z_",   "", clean)
  clean <- sub("^is_",  "", clean)
  clean <- gsub("_",    " ", clean)
  paste0(toupper(substr(clean, 1, 1)), substr(clean, 2, nchar(clean)))
}

# β LaTeX (sans $) depuis un nom de paramètre R.
# Interactions A:B → β_{\text{A} × \text{B}} (× hors \text{}).
.name_to_beta <- function(nm) {
  if (nm == "(Intercept)") return("\\mu")
  if (grepl(":", nm)) {
    parts  <- strsplit(nm, ":")[[1]]
    cleans <- sapply(parts, .id_text)
    return(paste0("\\beta_{",
                  paste(paste0("\\text{", cleans, "}"), collapse = " \\times "),
                  "}"))
  }
  paste0("\\beta_{\\text{", .id_text(nm), "}}")
}

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
  ifelse(p < 0.001, "",
  ifelse(p < 0.01,  "",
  ifelse(p < 0.05,  "", ""))))
}

.beta_label <- function(nm) paste0("$", .name_to_beta(nm), "$")

# ── Export LaTeX : tableau des paramètres ─────────────────────────────────────
# sigmas : liste de list(label=..., value=...) pour les écarts-types aléatoires
.params_to_latex <- function(params_df, model_name,
                              equation = NULL, sigmas = NULL) {
  skip_patterns <- c("^sd_", "^cor_", "^sigma$", "^Residual$")

  eq_block <- if (!is.null(equation)) {
    paste0("\\The final expression for the speed is \n", equation, "\n\n")
  } else ""

  lines <- c(
    eq_block,
    "\\begin{table}[h!]\\centering\\small",
    "\\begin{tabular}{lrrrr}",
    "\\hline\\hline",
    "Parameter & Value & Std. err. & $t$-stat. & $p$-value \\\\",
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

  # Bloc effets aléatoires (sigmas) en bas du tableau
  if (!is.null(sigmas) && length(sigmas) > 0) {
    lines <- c(lines, "\\hline")
    for (s in sigmas) {
      val_str <- if (is.na(s$value)) "---" else sprintf("%.4f km/h", s$value)
      lines <- c(lines,
        sprintf("%s & %s & --- & --- & --- \\\\", s$label, val_str))
    }
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

# ── Equation LaTeX du modèle ──────────────────────────────────────────────────
# Symbole de variable LaTeX : x̃ (z-score), 𝟙 (dummy factor), x (autre).
# Le texte du subscript vient de .id_text (ID_LABELS ou auto-génération).
.var_sym <- function(v) {
  text <- .id_text(v)
  if (grepl("^z_", v)) {
    paste0("\\tilde{x}_{\\text{", text, "}}")
  } else if (grepl("^(genre|time_of_day|experience|at_intersection|is_afternoon|WEATHER|LIGHTING|SURFACE|ZONE|VISUAL|RIDING|day_|season|month)", v)) {
    paste0("\\math{1}_{\\{\\text{", text, "}\\}}")
  } else {
    paste0("\\tilde{x}_{\\text{", text, "}}")
  }
}

.build_equation_latex <- function(params_df, mixed = FALSE, panel_cols = NULL) {
  skip_patterns <- c("^sd_", "^cor_", "^sigma$", "^Residual$")
  nms <- rownames(params_df)
  nms <- nms[!sapply(nms, function(n)
    any(sapply(skip_patterns, function(p) grepl(p, n))))]

  lhs <- if (mixed) "\\hat{y}_{it}" else "\\hat{y}"

  terms <- character(0)
  for (nm in nms) {
    blbl <- .name_to_beta(nm)
    if (nm == "(Intercept)") {
      terms <- c(terms, blbl)
    } else if (grepl(":", nm)) {
      parts <- strsplit(nm, ":")[[1]]
      xsym  <- paste(sapply(parts, .var_sym), collapse = " \\cdot ")
      terms <- c(terms, paste0(blbl, " \\cdot ", xsym))
    } else {
      terms <- c(terms, paste0(blbl, " \\cdot ", .var_sym(nm)))
    }
  }

  if (mixed) {
  # u_i
    terms <- c(terms, "u_i \\quad u_i \\sim \\mathcal{N}(0, \\sigma_{\\text{rider}}^2)")

    # autres effets aléatoires (u_j, u_k, ...)
    if (!is.null(panel_cols) && length(panel_cols) > 1) {
      for (idx in seq_along(panel_cols[-1])) {
        letter <- letters[idx + 9]  # j, k, ...
        lbl <- if (panel_cols[idx + 1] == "source") "trip" else panel_cols[idx + 1]
        terms <- c(terms,
          sprintf("u_%s \\quad u_%s \\sim \\mathcal{N}(0, \\sigma_{\\text{%s}}^2)",
                  letter, letter, lbl)
        )
      }
    }

    # epsilon
    terms <- c(terms,
      "\\varepsilon_{it} \\quad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma_{\\varepsilon}^2)"
    )
  } else {
    terms <- c(terms, "\\varepsilon")
  }

  lines <- c(
    "\\begin{align*}",
    paste0("  ", lhs, " &= ", terms[1], " \\\\")
  )
  for (t in terms[-c(1, length(terms))]) {
    lines <- c(lines, paste0("    &\\quad + ", t, " \\\\"))
  }
  lines <- c(lines,
    paste0("    &\\quad + ", terms[length(terms)]),
    "\\end{align*}"
  )
  paste(lines, collapse = "\n")
}

# ── Export LaTeX : tableau des statistiques du modèle ─────────────────────────
.stats_to_latex <- function(metrics, model_name) {
  lrt_p <- metrics$LRT_p
  sig   <- if (is.na(lrt_p)) "" else .sig_stars(lrt_p)

  # ── Comptages ────────────────────────────────────────────────────────────────
  rows <- list(
    c("$N_{\\text{obs}}$",    sprintf("%d", metrics$N)),
    c("$N_{\\text{riders}}$", if (!is.null(metrics$N_riders)) sprintf("%d", metrics$N_riders) else "---")
  )
  if (!is.null(metrics$extra_ns) && length(metrics$extra_ns) > 0) {
    for (pc in names(metrics$extra_ns)) {
      lbl <- if (pc == "source") "$N_{\\text{trips}}$" else sprintf("$N_{\\text{%s}}$", pc)
      rows <- c(rows, list(c(lbl, sprintf("%d", metrics$extra_ns[[pc]]))))
    }
  }

  # ── Log-vraisemblance et LRT global ─────────────────────────────────────────
  rows <- c(rows, list(
    c("$K$",                                       sprintf("%d",   metrics$K)),
    c("$\\mathcal{LL}(\\text{cst})$",              sprintf("%d",   round(metrics$LL_null))),
    c("$\\mathcal{LL}(\\hat{\\beta})$",            sprintf("%d",   round(metrics$LL_final))),
    c("%$\\bar{\\rho}^2$",                          sprintf("%.4f", metrics$rho2_bar)),
    c(sprintf("%% LRT $\\chi^2(%d)$ vs nul", metrics$LRT_df),
      sprintf("%% %.2f%s", metrics$LRT_stat, sig)),
    c("%% $p$-value LRT",
      if (is.na(lrt_p)) "%% ---" else sprintf("%% %.4f", lrt_p))
  ))

  # ── R² ───────────────────────────────────────────────────────────────────────
  # OLS : R² et R²_adj simples
  if (!is.null(metrics$r2)) {
    rows <- c(rows, list(
      c("$R^2$",        sprintf("%.4f", metrics$r2)),
      c("$\\bar{R}^2$", sprintf("%.4f", metrics$r2_adj))
    ))
  }
  # Mixte : Rm² et Rc² — formule complète (σ_rider, σ_trip, …) dans le label
  if (!is.null(metrics$r2_marginal)) {
    sig_parts <- c("\\sigma^2_f", "\\sigma^2_{\\text{rider}}")
    if (!is.null(metrics$extra_sigmas) && length(metrics$extra_sigmas) > 0) {
      for (pc in names(metrics$extra_sigmas)) {
        lbl <- if (pc == "source") "trip" else pc
        sig_parts <- c(sig_parts, sprintf("\\sigma^2_{\\text{%s}}", lbl))
      }
    }
    sig_parts <- c(sig_parts, "\\sigma^2_\\varepsilon")
    denom   <- paste(sig_parts, collapse = " + ")
    numer_c <- paste(sig_parts[-length(sig_parts)], collapse = " + ")
    rows <- c(rows, list(
      c(sprintf("$R^2_m = \\frac{\\sigma^2_f}{%s}$",  denom),
        sprintf("%.4f", metrics$r2_marginal)),
      c(sprintf("$R^2_c = \\frac{%s}{%s}$", numer_c, denom),
        sprintf("%.4f", metrics$r2_conditional))
    ))
  }

  # ── LRT effets aléatoires ────────────────────────────────────────────────────
  if (!is.null(metrics$sigma_rider)) {
    sigma_rid_p   <- if (!is.null(metrics$sigma_rider_p))   metrics$sigma_rider_p   else NA
    sigma_rid_lrt <- if (!is.null(metrics$sigma_rider_lrt)) metrics$sigma_rider_lrt else NA
    sig_re <- .sig_stars(sigma_rid_p)
    rows <- c(rows, list(
      c("LRT $\\chi^2(1)$ vs $\\sigma_{\\text{rider}}=0$",
        if (is.na(sigma_rid_lrt)) "---" else sprintf("%.2f%s", sigma_rid_lrt, sig_re)),
      c("$p$-value LRT $\\sigma_{\\text{rider}}=0$",
        if (is.na(sigma_rid_p)) "---" else sprintf("%.4f", sigma_rid_p))
    ))
    if (!is.null(metrics$extra_lrts) && length(metrics$extra_lrts) > 0) {
      for (pc in names(metrics$extra_lrts)) {
        lrt_info <- metrics$extra_lrts[[pc]]
        if (!is.null(lrt_info)) {
          sig_extra <- .sig_stars(lrt_info$p)
          lbl_lrt <- if (pc == "source") "LRT $\\chi^2(1)$ vs $\\sigma_{\\text{trip}}=0$"
                     else sprintf("LRT $\\chi^2(1)$ vs $\\sigma_{\\text{%s}}=0$", pc)
          lbl_p   <- if (pc == "source") "$p$-value LRT $\\sigma_{\\text{trip}}=0$"
                     else sprintf("$p$-value LRT $\\sigma_{\\text{%s}}=0$", pc)
          rows <- c(rows, list(
            c(lbl_lrt, if (is.na(lrt_info$lrt)) "---" else sprintf("%.2f%s", lrt_info$lrt, sig_extra)),
            c(lbl_p,   if (is.na(lrt_info$p))   "---" else sprintf("%.4f", lrt_info$p))
          ))
        }
      }
    }
  }

  lines <- c(
    "\\begin{table}[h!]\\centering\\small",
    "\\begin{tabular}{lr}",
    "\\hline\\hline",
    "Statistics & Value \\\\",
    "\\hline"
  )
  for (r in rows) lines <- c(lines, paste0(r[[1]], " & ", r[[2]], " \\\\"))
  lines <- c(lines,
    "\\hline\\hline",
    "\\end{tabular}",
    paste0("\\caption{Statistics of the linear mixed-effect model predicting the speed}"),
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
run_linear <- function(df_est, rhs, model_name, ref = NULL) {
  # ref : résultat de run_linear(..., "1", "M0") pour comparer au même modèle
  #       constant global. Si NULL, un modèle constant est estimé localement.

  formula_obj  <- as.formula(paste("speed_kmh_t1 ~", rhs))

  vars_used <- unique(c("speed_kmh_t1", all.vars(formula_obj)))
  vars_used <- vars_used[vars_used %in% names(df_est)]
  data      <- df_est[, vars_used, drop = FALSE]
  before    <- nrow(data)
  data      <- data[complete.cases(data), ]
  dropped   <- before - nrow(data)
  if (dropped > 0) message(sprintf("[%s] ⚠ %d lignes supprimées (NaN)", model_name, dropped))
  N <- nrow(data)

  # ── Estimation ────────────────────────────────────────────────────────────
  fit <- lm(formula_obj, data = data)

  ll <- as.numeric(logLik(fit))
  k  <- length(coef(fit)) + 1   # coefs + sigma

  # Modèle nul : externe (ref) ou estimé localement sur les mêmes données
  if (!is.null(ref)) {
    ll_null <- ref$metrics$LL_final
    k_null  <- ref$metrics$K
  } else {
    fit0    <- lm(speed_kmh_t1 ~ 1, data = data)
    ll_null <- as.numeric(logLik(fit0))
    k_null  <- 2   # mu + sigma
  }

  # ── Métriques ─────────────────────────────────────────────────────────────
  rho2     <- 1 - ll / ll_null
  rho2_bar <- 1 - (ll - k) / ll_null
  aic_val  <- AIC(fit)
  bic_val  <- BIC(fit)
  lrt_stat <- -2 * (ll_null - ll)
  lrt_df   <- k - k_null
  lrt_p    <- if (lrt_df > 0) pchisq(lrt_stat, df = lrt_df, lower.tail = FALSE) else NA

  r2_val     <- summary(fit)$r.squared
  r2_adj_val <- summary(fit)$adj.r.squared

  sig_str <- if (!is.na(lrt_p) && lrt_p < 0.001) " ***" else
             if (!is.na(lrt_p) && lrt_p < 0.01)  " **"  else
             if (!is.na(lrt_p) && lrt_p < 0.05)  " *"   else " (n.s.)"

  cat(sprintf("\n%s\n", strrep("=", 65)))
  cat(sprintf("  Modèle cst      : LL=%.2f\n", ll_null))
  cat(sprintf("  Modèle principal: N=%d  K=%d  LL=%.2f\n", N, k, ll))
  cat(sprintf("  R²=%.4f  R²_adj=%.4f\n", r2_val, r2_adj_val))
  cat(sprintf("  rho²=%.4f  AIC=%.1f  BIC=%.1f\n", rho2, aic_val, bic_val))
  cat(sprintf("  LRT vs cst : chi²(%d)=%.2f  p=%.4f%s\n",
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
    r2       = round(r2_val,     4),
    r2_adj   = round(r2_adj_val, 4),
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
  writeLines(.params_to_latex(params_df, model_name,
                              equation = .build_equation_latex(params_df, mixed = FALSE)),
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
                                   panel_id_col = "rider_id",
                                   method = "ML") {
  # method : "ML"   = maximum likelihood   (REML=FALSE, LRT sur effets fixes valide)
  #          "REML" = restricted likelihood (REML=TRUE,  LRT sur effets fixes invalide)
  #          Les deux méthodes utilisent REML=FALSE pour le LRT (fit ML interne).
  use_reml <- identical(toupper(method), "REML")

  # panel_id_col peut être un vecteur de colonnes, ex. c("rider_id", "source")
  panel_cols <- panel_id_col

  # Construction des termes aléatoires : (1|col1) + (1|col2) + ...
  re_terms <- paste(sprintf("(1|%s)", panel_cols), collapse = " + ")

  formula_obj  <- as.formula(paste0("speed_kmh_t1 ~ ", rhs, " + ", re_terms))
  formula_null <- as.formula(paste0("speed_kmh_t1 ~ ", re_terms))

  vars_used <- unique(c("speed_kmh_t1", panel_cols,
                         all.vars(as.formula(paste("~", rhs)))))
  vars_used <- vars_used[vars_used %in% names(df_est)]
  data      <- df_est[, vars_used, drop = FALSE]
  before    <- nrow(data)
  data      <- data[complete.cases(data), ]
  dropped   <- before - nrow(data)
  if (dropped > 0) message(sprintf("[%s] ⚠ %d lignes supprimées (NaN)", model_name, dropped))

  # ── Vérification multicolinéarité (|r| > 0.7) ────────────────────────────
  pred_vars <- setdiff(all.vars(as.formula(paste("~", rhs))), names(data)[!names(data) %in% names(df_est)])
  pred_vars <- intersect(pred_vars, names(data))
  num_vars  <- pred_vars[sapply(data[pred_vars], is.numeric)]
  if (length(num_vars) >= 2) {
    cor_mat <- cor(data[num_vars], use = "complete.obs")
    cor_mat[lower.tri(cor_mat, diag = TRUE)] <- NA
    idx     <- which(!is.na(cor_mat), arr.ind = TRUE)
    cor_pairs <- data.frame(
      var1 = rownames(cor_mat)[idx[, 1]],
      var2 = colnames(cor_mat)[idx[, 2]],
      r    = cor_mat[idx],
      stringsAsFactors = FALSE
    )
    cor_pairs <- cor_pairs[order(abs(cor_pairs$r)), ]
    cat(sprintf("\n  [%s] Corrélations entre prédicteurs (ordre croissant |r|) :\n", model_name))
    for (i in seq_len(nrow(cor_pairs))) {
      flag <- if (abs(cor_pairs$r[i]) > 0.7) "  ⚠ > 0.7" else ""
      cat(sprintf("    cor(%-30s, %-30s) = %+.3f%s\n",
                  cor_pairs$var1[i], cor_pairs$var2[i], cor_pairs$r[i], flag))
    }
    cat("\n")
  }

  # Trier par premier identifiant panel
  data <- data[order(data[[panel_cols[1]]]), ]

  N_obs    <- nrow(data)
  N_riders <- length(unique(data[[panel_cols[1]]]))

  # ── Estimation principale ─────────────────────────────────────────────────
  fit  <- lmer(formula_obj,  data = data, REML = use_reml)
  if (use_reml) {
    message(sprintf("[%s] ℹ REML=TRUE : re-estimation ML interne pour LRT sur effets fixes.", model_name))
    fit_ml  <- lmer(formula_obj,  data = data, REML = FALSE)
    fit0_ml <- lmer(formula_null, data = data, REML = FALSE)
  } else {
    fit_ml  <- fit
    fit0_ml <- lmer(formula_null, data = data, REML = FALSE)
  }
  fit0 <- fit0_ml   # nul toujours en ML pour LRT

  ll      <- as.numeric(logLik(fit_ml))
  ll_null <- as.numeric(logLik(fit0_ml))   # nul lmer (RE seuls) → LRT
  k       <- attr(logLik(fit_ml),  "df")
  k_null  <- attr(logLik(fit0_ml), "df")

  # Nul OLS pur (μ seul, sans RE) → ρ² comparable à run_linear
  ll_null_ols <- as.numeric(logLik(lm(speed_kmh_t1 ~ 1, data = data)))
  k_null_ols  <- 2L   # μ + σ

  # ── Métriques ─────────────────────────────────────────────────────────────
  rho2     <- 1 - ll / ll_null_ols
  rho2_bar <- 1 - (ll - k) / ll_null_ols
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
  sigma_rid <- sqrt(re_df$vcov[re_df$grp == panel_cols[1]])
  sigma_eps <- sqrt(re_df$vcov[re_df$grp == "Residual"])

  icc <- sigma_rid^2 / (sigma_rid^2 + sigma_eps^2)

  # ── R² marginal et conditionnel (MuMIn::r.squaredGLMM, Nakagawa & Schielzeth 2013) ──
  r2_mumin       <- r.squaredGLMM(fit)
  r2_marginal    <- r2_mumin[1, "R2m"]
  r2_conditional <- r2_mumin[1, "R2c"]

  cat(sprintf("\n%s\n", strrep("=", 72)))
  cat(sprintf("  Mixed panel linear model: %s  [method: %s]\n", model_name, toupper(method)))
  cat(sprintf("  Panel: %s\n", paste(panel_cols, collapse = " + ")))
  cat(sprintf("  Riders=%d  Observations=%d\n", N_riders, N_obs))
  cat(sprintf("  sigma_%s=%.4f  sigma_eps=%.4f\n", panel_cols[1], sigma_rid, sigma_eps))
  cat(sprintf("  Modèle nul      : LL=%.2f\n", ll_null))
  cat(sprintf("  Modèle principal: K=%d  LL=%.2f\n", k, ll))
  cat(sprintf("  Rm²=%.4f  Rc²=%.4f\n", r2_marginal, r2_conditional))
  cat(sprintf("  rho²=%.4f  AIC=%.1f  BIC=%.1f\n", rho2, aic_val, bic_val))
  cat(sprintf("  LRT vs nul : chi²(%d)=%.2f  p=%.4f%s\n",
              lrt_df, lrt_stat, lrt_p, sig_str))
  cat(sprintf("%s\n", strrep("=", 72)))
  print(summary(fit)$coefficients)

  # Effets aléatoires supplémentaires (panel cols 2+) + LRT via ranova
  extra_sigmas <- list()
  extra_ns     <- list()
  extra_lrts   <- list()   # list(lrt=..., p=...)
  if (length(panel_cols) > 1) {
    ranova_res_all <- ranova(fit)
    for (pc in panel_cols[-1]) {
      sig_val <- re_df$vcov[re_df$grp == pc]
      extra_sigmas[[pc]] <- if (length(sig_val) > 0) round(sqrt(sig_val), 4) else NA
      extra_ns[[pc]]     <- length(unique(data[[pc]]))
      row_idx <- grep(pc, rownames(ranova_res_all), value = FALSE)[1]
      if (!is.na(row_idx)) {
        extra_lrts[[pc]] <- list(
          lrt = round(ranova_res_all[["LRT"]][row_idx], 2),
          p   = ranova_res_all[["Pr(>Chisq)"]][row_idx]
        )
      } else {
        extra_lrts[[pc]] <- list(lrt = NA, p = NA)
      }
    }
  }

  # p-value pour sigma du 1er panel col via ranova (LRT sigma=0)
  ranova_res      <- ranova(fit)
  ranova_row      <- grep(panel_cols[1], rownames(ranova_res), value = FALSE)[1]
  if (!is.na(ranova_row)) {
    sigma_rider_lrt <- ranova_res[["LRT"]][ranova_row]
    sigma_rider_p   <- ranova_res[["Pr(>Chisq)"]][ranova_row]
  } else {
    sigma_rider_lrt <- NA
    sigma_rider_p   <- NA
  }

  metrics <- list(
    Model         = model_name,
    N             = N_obs,
    N_riders      = N_riders,
    K             = k,
    LL_null       = round(ll_null_ols,   2),   # μ OLS pur (base du ρ²)
    LL_final      = round(ll,           2),
    rho2          = round(rho2,         4),
    rho2_bar      = round(rho2_bar,     4),
    AIC           = round(aic_val,      2),
    BIC           = round(bic_val,      2),
    LRT_stat      = round(lrt_stat,     2),
    LRT_df        = lrt_df,
    LRT_p         = if (!is.na(lrt_p)) round(lrt_p, 4) else NA,
    r2_marginal    = round(r2_marginal,    4),
    r2_conditional = round(r2_conditional, 4),
    sigma_rider   = round(sigma_rid,    4),
    sigma_eps     = round(sigma_eps,    4),
    ICC           = round(icc,          4),
    sigma_rider_lrt = if (!is.na(sigma_rider_lrt)) round(sigma_rider_lrt, 2) else NA,
    sigma_rider_p   = sigma_rider_p,
    extra_sigmas  = extra_sigmas,
    extra_ns      = extra_ns,
    extra_lrts    = extra_lrts
  )

  # ── Sauvegarde ────────────────────────────────────────────────────────────
  out_dir   <- .get_out_dir(model_name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  params_df <- as.data.frame(summary(fit)$coefficients)
  write.csv(params_df, file.path(out_dir, paste0(model_name, "_params.csv")))

  # Construire la liste des sigmas pour params.tex
  sigmas_list <- list(
    list(label = "$\\sigma_{\\text{rider}}$", value = metrics$sigma_rider),
    list(label = "$\\sigma_{\\varepsilon}$",  value = metrics$sigma_eps)
  )
  if (!is.null(metrics$extra_sigmas) && length(metrics$extra_sigmas) > 0) {
    for (pc in names(metrics$extra_sigmas)) {
      lbl <- if (pc == "source") "$\\sigma_{\\text{trip}}$" else
             sprintf("$\\sigma_{\\text{%s}}$", pc)
      sigmas_list <- c(sigmas_list,
        list(list(label = lbl, value = metrics$extra_sigmas[[pc]])))
    }
  }

  writeLines(.params_to_latex(params_df, model_name,
                              equation = .build_equation_latex(params_df, mixed = TRUE, panel_cols = panel_cols),
                              sigmas   = sigmas_list),
             file.path(out_dir, paste0(model_name, "_params.tex")))
  writeLines(.stats_to_latex(metrics, model_name),
             file.path(out_dir, paste0(model_name, "_stats.tex")))

  invisible(list(fit = fit, params = params_df, metrics = metrics))
}

cat("✔ Fonctions R chargées : run_linear, run_mixed_linear_panel\n")
