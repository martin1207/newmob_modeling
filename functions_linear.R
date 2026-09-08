# ══════════════════════════════════════════════════════════════════════════════
# functions_linear.R
# Equivalent R de funtion_linear.py
# Packages requis : nlme, MuMIn, clubSandwich
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  # lme4/lmerTest ne sont plus utilisés : tous les modèles mixtes passent par nlme
  # (run_mixed_linear_panel délègue à run_mixed_linear_panel_ar avec ar_order = 0).
  if (!requireNamespace("MuMIn",     quietly = TRUE)) install.packages("MuMIn",     repos = "https://cloud.r-project.org")
  if (!requireNamespace("showtext",  quietly = TRUE)) install.packages("showtext",  repos = "https://cloud.r-project.org")
  if (!requireNamespace("sysfonts",  quietly = TRUE)) install.packages("sysfonts",  repos = "https://cloud.r-project.org")
  if (!requireNamespace("ggeffects", quietly = TRUE)) install.packages("ggeffects", repos = "https://cloud.r-project.org")
  library(MuMIn)      # r.squaredGLMM : R² marginal et conditionnel
  library(nlme)       # lme (+ corARMA) : tous les modèles mixtes
})

# ── Labels LaTeX ──────────────────────────────────────────────────────────────
# Seule table de labels : variable R → texte d'affichage (sans LaTeX).
# Utilisée à la fois pour les β et pour les symboles de variables (x̃, 𝟙, x).
ID_LABELS <- c(
  "z_n_pedestrians"            = "N visible pedestrians < 15m",
  "z_n_cyclists"              = "N visible cyclists < 15m",
  "z_n_vru_ped_cyc"           = "N visible pedestrians + cyclists < 15m",

  "z_road_width_perp_m"        = "Road width (m)",
  "road_width_catmedium"       = "Medium road (6–12m, ref: wide)",
  "road_width_catnarrow"       = "Narrow road (<6m, ref: wide)",
  "z_n_elderly"                = "N elderly",
  "z_n_children"               = "N children",
  "z_n_running"                = "N visible running pedestrians < 15m",
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
  "z_prop_interaction_same_direction" = "Proportion of same-direction encounters",
  "z_prop_interaction_opposite_direction" = "Proportion of opposite-direction encounters",
  "z_prop_interaction_crossing"         = "Proportion of crossing encounters",
  "is_afternoon"                        = "Afternoon (ref : Morning)",
  "is_park"                             = "Park (ref : other)",
  "is_square"                           = "Square (ref : other)"
)

# ── Ordre d'affichage préféré pour les graphes marginaux ─────────────────────
PLOT_VAR_ORDER <- c(
  "genremale", "genrefemale",
  "z_prop_interaction_same_direction",
  "z_n_cyclists",
  "z_n_pedestrians",
  "z_n_running",
  "is_afternoon",
  "at_intersection", "at_intersection1"
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
  if (nm == "z_speed_kmh_kalman") return("\\theta")
  if (nm == "z_speed_kmh_kalman_t0") return("\\b_0")

  if (grepl(":", nm)) {
    parts  <- strsplit(nm, ":")[[1]]
    cleans <- sapply(parts, .id_text)
    return(paste0("\\beta_{",
                  paste(paste0("\\text{", cleans, "}"), collapse = " \\times "),
                  "}"))
  }
  paste0("\\beta_{\\text{", .id_text(nm), "}}")
}

# ── SE cluster-robustes CR2 (clubSandwich) au format params_df ────────────────
# Renvoie un data.frame avec mêmes colonnes que summary()$coefficients
# (Estimate / Std. Error / df / t value / Pr(>|t|)) mais SE/t/p cluster-robustes
# CR2 + dof de Satterthwaite, clusterisées par `cluster_col`. Estimations
# inchangées. Repli (avec warning) sur le modèle si clubSandwich indisponible.
.robust_params_df <- function(fit, cluster_col = "rider_id") {
  if (!requireNamespace("clubSandwich", quietly = TRUE)) {
    warning("clubSandwich absent → colonnes robustes omises"); return(NULL)
  }
  d  <- tryCatch(nlme::getData(fit), error = function(e) NULL)
  if (is.null(d)) d <- tryCatch(model.frame(fit), error = function(e) NULL)
  cl <- if (!is.null(d) && cluster_col %in% names(d)) d[[cluster_col]] else NULL
  ct <- tryCatch(
    clubSandwich::coef_test(fit, vcov = "CR2", cluster = cl, test = "Satterthwaite"),
    error = function(e) { warning(sprintf("coef_test CR2 échoué (%s) → colonnes robustes omises",
                                          conditionMessage(e))); NULL })
  if (is.null(ct)) return(NULL)
  tab <- as.data.frame(ct)
  nm  <- if ("Coef" %in% names(tab)) as.character(tab$Coef) else rownames(tab)
  dfv <- if ("df_Satt" %in% names(tab)) tab$df_Satt else tab$df
  pv  <- if ("p_Satt" %in% names(tab)) tab$p_Satt else if ("p_val" %in% names(tab)) tab$p_val else NA
  out <- data.frame(Estimate = tab$beta, `Std. Error` = tab$SE, df = dfv,
                    `t value` = tab$tstat, `Pr(>|t|)` = pv,
                    check.names = FALSE, row.names = nm)
  attr(out, "n_clusters") <- length(unique(cl)); attr(out, "cluster_col") <- cluster_col
  out
}

# Fusionne les colonnes robustes (SE_CR2, t_CR2, df_CR2, p_CR2) dans le CSV params.
.merge_robust_csv <- function(params_df, robust_df) {
  out <- as.data.frame(params_df)
  if (!is.null(robust_df)) {
    m <- match(rownames(out), rownames(robust_df))
    out[["SE_CR2"]] <- robust_df[["Std. Error"]][m]
    out[["t_CR2"]]  <- robust_df[["t value"]][m]
    out[["df_CR2"]] <- robust_df[["df"]][m]
    out[["p_CR2"]]  <- robust_df[["Pr(>|t|)"]][m]
  }
  out
}

# ── Utilitaires internes ──────────────────────────────────────────────────────
.get_out_dir <- function(model_name) {
  base <- file.path("model_results_linear", model_name)
  if (!dir.exists(base)) return(base)
  i <- 2
  while (dir.exists(paste0(base, "_v", i))) i <- i + 1
  paste0(base, "_v", i)
}

# ── Marginal means plot ───────────────────────────────────────────────────────
# Produit un graphique par variable significative (p < alpha) :
#   • Variables z_* (continues standardisées) : courbe prédite sur [-2, 2] SD,
#     axe x re-transformé en unités originales (z * sd + mean)
#   • Autres variables numériques             : courbe sur [min, max]
#   • Variables binaires / facteurs           : points par modalité
# Les autres prédicteurs sont fixés à leur moyenne (0 pour les z_vars).
# Sauvegarde : <out_dir>/<model_name>_marginal_<varname>.pdf
.plot_marginal_means <- function(fit, params_df, data, model_name,
                                 out_dir, alpha = 0.05, is_mixed = FALSE,
                                 raw_data = NULL) {
  # raw_data : df_est complet (toutes colonnes), utilisé pour récupérer les
  # colonnes originales (non standardisées) des z_* variables
  if (is.null(raw_data)) raw_data <- data

  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("ggplot2 requis pour .plot_marginal_means")

  # ── Police LaTeX (EB Garamond via showtext) ───────────────────────────────
  use_latex_font <- FALSE
  if (requireNamespace("showtext", quietly = TRUE) &&
      requireNamespace("sysfonts", quietly = TRUE)) {
    tryCatch({
      sysfonts::font_add_google("EB Garamond", "ebgaramond")
      showtext::showtext_auto()
      use_latex_font <- TRUE
    }, error = function(e) {
      message("showtext: police EB Garamond non chargée — police système utilisée")
    })
  }
  base_family <- if (use_latex_font) "ebgaramond" else "serif"

  # ── Thème publication ────────────────────────────────────────────────────
  theme_latex <- ggplot2::theme_bw(base_size = 16, base_family = base_family) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(size = 18, face = "bold",
                                               margin = ggplot2::margin(b = 4)),
      plot.subtitle    = ggplot2::element_text(size = 14, color = "grey30",
                                               margin = ggplot2::margin(b = 8)),
      axis.title       = ggplot2::element_text(size = 16),
      axis.text        = ggplot2::element_text(size = 14, color = "grey20"),
      panel.grid.major = ggplot2::element_line(color = "grey90", linewidth = 0.4),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border     = ggplot2::element_rect(color = "grey60", linewidth = 0.6),
      plot.margin      = ggplot2::margin(8, 12, 8, 8)
    )

  # p-value column name varies between lm and lmer outputs
  pvcol <- intersect(c("Pr(>|t|)", "Pr(>|z|)"), colnames(params_df))[1]
  if (is.na(pvcol)) return(invisible(NULL))

  # Significant fixed effects (exclude intercept)
  sig_rows <- rownames(params_df)[
    !grepl("Intercept", rownames(params_df)) 
  ]
  if (length(sig_rows) == 0) {
    message(sprintf("[%s] Aucune variable significative (alpha=%.2f) — pas de graphique.", model_name, alpha))
    return(invisible(NULL))
  }

  # Fixed-effects vcov & design matrix helper
  vc_fix <- if (is_mixed) as.matrix(vcov(fit)) else vcov(fit)

  # Build baseline row: means of numeric cols, ref level for factors
  num_cols  <- names(data)[sapply(data, is.numeric)]
  fac_cols  <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
  baseline  <- as.data.frame(lapply(names(data), function(nm) {
    if (nm %in% num_cols) mean(data[[nm]], na.rm = TRUE)
    else                  data[[nm]][1]          # first obs = reference
  }), stringsAsFactors = FALSE)
  names(baseline) <- names(data)

  # Force z_* baseline to 0 (standardised → mean = 0)
  z_cols <- grep("^z_", names(baseline), value = TRUE)
  for (zc in z_cols) if (zc %in% names(baseline)) baseline[[zc]] <- 0

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("Installation de patchwork...")
    install.packages("patchwork", repos = "https://cloud.r-project.org")
  }

  plot_list <- list()   # collecte tous les ggplots

  for (var in sig_rows) {

    # Identify the raw column name :
    #   1. z_foo  → foo  (cherche dans raw_data pour avoir la colonne originale)
    #   2. exact match in data
    #   3. factor dummy : "at_intersection1" → "at_intersection" (longest prefix match)
    raw_col <- if (grepl("^z_", var) && sub("^z_", "", var) %in% names(raw_data))
                 sub("^z_", "", var)
               else if (var %in% names(data)) var
               else {
                 candidates <- names(data)[sapply(names(data), function(nm)
                   startsWith(var, nm) && nchar(var) > nchar(nm))]
                 if (length(candidates) > 0)
                   candidates[which.max(nchar(candidates))]
                 else NA_character_
               }

    ref_col_data  <- if (!is.na(raw_col) && raw_col %in% names(raw_data)) raw_data[[raw_col]]
                     else if (!is.na(raw_col) && raw_col %in% names(data)) data[[raw_col]]
                     else NULL
    is_binary     <- !is.null(ref_col_data) &&
                     all(ref_col_data %in% c(0, 1, NA), na.rm = TRUE)
    is_factor_var <- !is.null(ref_col_data) &&
                     (is.factor(ref_col_data) || is.character(ref_col_data))

    # ── Build prediction grid ──────────────────────────────────────────────
    x_display <- NULL   # sera défini dans le bloc continu si nécessaire
    # colonne à utiliser dans le grid : toujours raw_col si disponible dans data
    col_in_data <- if (!is.na(raw_col) && raw_col %in% names(data)) raw_col else var
    if (is_binary || is_factor_var) {
      levs <- if (is_factor_var) sort(unique(as.character(ref_col_data))) else c(0, 1)

      # Récupérer les levels exacts depuis les données du modèle
      col_levels <- if (col_in_data %in% names(data) && is.factor(data[[col_in_data]]))
                      levels(data[[col_in_data]])
                    else if (!is.null(ref_col_data))
                      sort(unique(as.character(ref_col_data)))
                    else levs

      grid <- do.call(rbind, lapply(levs, function(lv) {
        row <- baseline
        row[[col_in_data]] <- factor(lv, levels = col_levels)
        row
      }))
      x_var  <- col_in_data
      x_lab  <- .id_text(var)
      is_cat <- TRUE
    } else {
      n_pts <- 60

      is_z_var <- grepl("^z_", var) && !is.na(raw_col) && raw_col %in% names(raw_data)

      if (is_z_var) {
        # Plage = range complet de la variable originale depuis raw_data
        raw_mean  <- mean(raw_data[[raw_col]], na.rm = TRUE)
        raw_sd    <- sd(raw_data[[raw_col]],   na.rm = TRUE)
        rng_orig  <- range(raw_data[[raw_col]], na.rm = TRUE)
        orig_vals <- seq(rng_orig[1], rng_orig[2], length.out = n_pts)
        z_vals    <- (orig_vals - raw_mean) / raw_sd
        x_display <- orig_vals
        x_lab     <- .id_text(var)
        grid <- do.call(rbind, lapply(z_vals, function(v) {
          row <- baseline
          row[[var]] <- v
          row
        }))
      } else {
        col_for_range <- if (!is.na(raw_col) && raw_col %in% names(data)) raw_col else
                         if (var %in% names(data)) var else NA_character_
        if (is.na(col_for_range) || !any(is.finite(data[[col_for_range]]))) {
          message(sprintf("[%s] ⚠ '%s' : plage non finie — graphique ignoré.", model_name, var))
          next
        }
        orig_vals <- seq(min(data[[col_for_range]], na.rm = TRUE),
                         max(data[[col_for_range]], na.rm = TRUE),
                         length.out = n_pts)
        x_display <- orig_vals
        x_lab     <- .id_text(var)
        grid <- do.call(rbind, lapply(orig_vals, function(v) {
          row <- baseline
          row[[var]] <- v
          row
        }))
      }
      x_var  <- var
      is_cat <- FALSE
    }

    # ── Prédiction + IC (méthode delta) + IP (ggeffects) ────────────────────────
      tryCatch({
        if (is_mixed) {
          # nlme::lme → prédiction population via level = 0 ; lme4 → re.form = NA
          if (inherits(fit, "lme")) {
            pred_vals <- predict(fit, newdata = grid, level = 0)
          } else {
            pred_vals <- predict(fit, newdata = grid, re.form = NA)
          }
        } else {
          pred_vals <- predict(fit, newdata = grid)
        }

        # IC via méthode delta (effets fixes uniquement)
        fx_form <- if (inherits(fit, "lme")) formula(fit) else formula(fit, fixed.only = TRUE)
        X       <- model.matrix(fx_form, data = grid)
        vc_fix  <- as.matrix(vcov(fit))
        common  <- intersect(colnames(X), colnames(vc_fix))
        X_sub   <- X[, common, drop = FALSE]
        vc_sub  <- vc_fix[common, common, drop = FALSE]
        var_ci  <- pmax(0, diag(X_sub %*% vc_sub %*% t(X_sub)))
        ci_lo   <- pred_vals - 1.96 * sqrt(var_ci)
        ci_hi   <- pred_vals + 1.96 * sqrt(var_ci)

        # IP pour une nouvelle observation d'un nouveau groupe (rider + trip) :
        #   se_PI = sqrt(var_IC_fixes + σ²_ε + σ²_rider + σ²_trip + ...)
        # Composants aléatoires : StdDev nlme ; repli lme4 si un merMod est passé
        # de l'extérieur (plus produit par ce fichier).
        if (is_mixed) {
          var_resid <- sigma(fit)^2
          if (inherits(fit, "lme")) {
            # nlme : StdDev des effets aléatoires = toutes les valeurs sauf le résidu
            sds       <- suppressWarnings(as.numeric(nlme::VarCorr(fit)[, "StdDev"]))
            sds       <- sds[!is.na(sds)]
            var_ranef <- if (length(sds) > 1) sum(head(sds, -1)^2) else 0
          } else {
            var_ranef <- if (requireNamespace("lme4", quietly = TRUE))
              sum(sapply(lme4::VarCorr(fit),
                         function(vc) {
                           v <- diag(as.matrix(vc))
                           sum(v[is.finite(v) & v > 0])
                         })) else 0
          }
          se_pi <- sqrt(var_ci + var_resid + var_ranef)
        } else {
          pred_pi <- predict(fit, newdata = grid,
                             interval = "prediction", level = 0.95)
          se_pi   <- (pred_pi[, "upr"] - pred_vals) / 1.96
        }
        pi_lo  <- pred_vals - 1.96 * se_pi
        pi_hi  <- pred_vals + 1.96 * se_pi
        has_ci <- TRUE

        # Valeurs axe x
        x_plot <- if (!is_cat && !is.null(x_display)) x_display else grid[[x_var]]

        plot_df <- data.frame(
          x      = x_plot,
          fit    = pred_vals,
          ci_lwr = ci_lo,
          ci_upr = ci_hi,
          pi_lwr = pi_lo,
          pi_upr = pi_hi
        )

      # ── Graphique ──────────────────────────────────────────────────────────
      if (is_cat) {
        g <- ggplot2::ggplot(plot_df, ggplot2::aes(x = factor(x), y = fit)) +
          { if (has_ci)
              ggplot2::geom_errorbar(ggplot2::aes(ymin = pi_lwr, ymax = pi_upr),
                                     width = 0.25, color = "#6BA3C8", linewidth = 1.0, alpha = 0.5)
          } +
          { if (has_ci)
              ggplot2::geom_errorbar(ggplot2::aes(ymin = ci_lwr, ymax = ci_upr),
                                     width = 0.12, color = "#1A3A5C", linewidth = 0.7, alpha = 0.9)
          } +
          ggplot2::geom_point(size = 3.5, color = "#1A3A5C") +
          ggplot2::labs(title = NULL, subtitle = NULL,
                        x = x_lab, y = "Speed (km/h)",
                        caption = NULL) +
          ggplot2::coord_cartesian(ylim = c(0, 30)) +
          theme_latex
      } else {
        g <- ggplot2::ggplot(plot_df, ggplot2::aes(x = x, y = fit)) +
          { if (has_ci)
              ggplot2::geom_ribbon(ggplot2::aes(ymin = pi_lwr, ymax = pi_upr),
                                   fill = "#6BA3C8", alpha = 0.12)
          } +
          { if (has_ci)
              ggplot2::geom_ribbon(ggplot2::aes(ymin = ci_lwr, ymax = ci_upr),
                                   fill = "#1A3A5C", alpha = 0.25)
          } +
          ggplot2::geom_line(color = "#1A3A5C", linewidth = 1.1) +
          ggplot2::scale_x_continuous(n.breaks = 6) +
          ggplot2::scale_y_continuous(n.breaks = 6) +
          ggplot2::labs(title = NULL, subtitle = NULL,
                        x = x_lab, y = "Speed (km/h)",
                        caption = NULL) +
          ggplot2::coord_cartesian(ylim = c(0, 30)) +
          theme_latex
      }

      plot_list[[var]] <- g

    }, error = function(e) {
      message(sprintf("[%s] ⚠ Impossible de tracer '%s' : %s", model_name, var, conditionMessage(e)))
    })
  }

  if (length(plot_list) == 0) return(invisible(NULL))

  # ── Tri selon PLOT_VAR_ORDER (variables non listées vont à la fin) ───────────
  ordered_keys <- c(
    intersect(PLOT_VAR_ORDER, names(plot_list)),          # dans l'ordre voulu
    setdiff(names(plot_list), PLOT_VAR_ORDER)             # reste alphabétique
  )
  plot_list <- plot_list[ordered_keys]

  # ── Assemblage en une seule figure ──────────────────────────────────────────
  n_plots <- length(plot_list)
  ncols   <- min(3L, n_plots)
  nrows   <- ceiling(n_plots / ncols)

  combined <- patchwork::wrap_plots(plot_list, ncol = ncols)

  out_file <- file.path(out_dir, paste0(model_name, "_marginal_all.png"))
  ggplot2::ggsave(out_file, combined,
                  width  = ncols * 4.5,
                  height = nrows * 3.5,
                  device = "png", dpi = 300)
  message(sprintf("[%s] ✔ marginal plots → %s", model_name, basename(out_file)))

  invisible(NULL)
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
                              equation = NULL, sigmas = NULL, robust_df = NULL) {
  skip_patterns <- c("^sd_", "^cor_", "^sigma$", "^Residual$")
  has_rob <- !is.null(robust_df)

  eq_block <- if (!is.null(equation)) {
    paste0("\\ The final expression for the speed is \n", equation, "\n\n")
  } else ""

  lines <- c(
    eq_block,
    "\\begin{table}[h!]\\centering\\small",
    paste0("\\begin{tabular}{", if (has_rob) "lrrrrrr" else "lrrrr", "}"),
    "\\hline\\hline",
    if (has_rob)
      "Parameter & Value & Std. err. & $t$-stat. & $p$-value & SE$_{\\text{CR2}}$ & $p_{\\text{CR2}}$ \\\\"
    else
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
    if (has_rob) {
      rse <- if (nm %in% rownames(robust_df)) robust_df[nm, "Std. Error"] else NA
      rpv <- if (nm %in% rownames(robust_df)) robust_df[nm, "Pr(>|t|)"]   else NA
      rse_str <- if (is.na(rse)) "---" else sprintf("%.4f", rse)
      rp_str  <- if (is.na(rpv)) "---" else sprintf("%.4f%s", rpv, .sig_stars(rpv))
      lines <- c(lines,
        sprintf("%s & %.4f & %.4f & %.3f & %s%s & %s & %s \\\\",
                .beta_label(nm), est, se, tv, pstr, sig, rse_str, rp_str))
    } else {
      lines <- c(lines,
        sprintf("%s & %.4f & %.4f & %.3f & %s%s \\\\",
                .beta_label(nm), est, se, tv, pstr, sig))
    }
  }

  # Bloc effets aléatoires (sigmas) en bas du tableau
  if (!is.null(sigmas) && length(sigmas) > 0) {
    lines <- c(lines, "\\hline")
    for (s in sigmas) {
      val_str <- if (is.na(s$value)) "---" else sprintf("%.4f km/h", s$value)
      tail_d  <- if (has_rob) "--- & --- & --- & --- & ---" else "--- & --- & ---"
      lines <- c(lines, sprintf("%s & %s & %s \\\\", s$label, val_str, tail_d))
    }
  }

  lines <- c(lines,
    "\\hline\\hline",
    "\\end{tabular}",
    paste0("\\caption{Estimated parameters of the linear regression",
           " model predicting the speed",
           if (has_rob) " (model-based SE and cluster-robust CR2 SE, clustered by rider)" else "",
           ".}"),
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

  # ── Corrélation temporelle AR(p) (modèles nlme::lme + corARMA) ───────────────
  if (!is.null(metrics$phi)) {
    ord <- if (!is.null(metrics$ar_order)) metrics$ar_order else length(metrics$phi)
    for (i in seq_along(metrics$phi)) {
      rows <- c(rows, list(c(
        sprintf("$\\varphi_{%d}$ (AR(%d))", i, ord),
        sprintf("%.4f", metrics$phi[i]))))
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
#   df_est     : data.frame avec au moins speed_kmh_kalman_t1 + les variables de rhs
#   rhs        : partie droite de la formule, ex. "z_n_pedestrians + genre_female"
#                Utilisez "1" pour le modèle nul (intercept seul).
#   model_name : identifiant du modèle (ex. "M1_pedestrians")
#
# Retour : liste(fit, params, metrics)
# ══════════════════════════════════════════════════════════════════════════════
run_linear <- function(df_est, rhs, model_name, ref = NULL) {
  # ref : résultat de run_linear(..., "1", "M0") pour comparer au même modèle
  #       constant global. Si NULL, un modèle constant est estimé localement.

  formula_obj  <- as.formula(paste("speed_kmh_kalman_t1 ~", rhs))

  vars_used <- unique(c("speed_kmh_kalman_t1", all.vars(formula_obj)))
  vars_used <- vars_used[vars_used %in% names(df_est)]
  data      <- df_est[, vars_used, drop = FALSE]
  for (cn in names(data)) {
    if (is.character(data[[cn]])) data[[cn]] <- factor(data[[cn]])
  }
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
    fit0    <- lm(speed_kmh_kalman_t1 ~ 1, data = data)
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

  # ── Corrélations entre betas estimés (cov2cor sur vcov des effets fixes) ──
  cor_beta <- cov2cor(as.matrix(vcov(fit)))
  cor_beta[lower.tri(cor_beta, diag = TRUE)] <- NA
  idx_b <- which(!is.na(cor_beta), arr.ind = TRUE)
  if (nrow(idx_b) > 0) {
    cor_beta_pairs <- data.frame(
      var1 = rownames(cor_beta)[idx_b[, 1]],
      var2 = colnames(cor_beta)[idx_b[, 2]],
      r    = cor_beta[idx_b],
      stringsAsFactors = FALSE
    )
    cor_beta_pairs <- cor_beta_pairs[order(abs(cor_beta_pairs$r), decreasing = TRUE), ]
    cat(sprintf("\n  [%s] Corrélations entre betas estimés (ordre décroissant |r|) :\n", model_name))
    for (i in seq_len(nrow(cor_beta_pairs))) {
      flag <- if (abs(cor_beta_pairs$r[i]) > 0.7) "  ⚠ > 0.7" else ""
      cat(sprintf("    cor(%-30s, %-30s) = %+.3f%s\n",
                  cor_beta_pairs$var1[i], cor_beta_pairs$var2[i], cor_beta_pairs$r[i], flag))
    }
    cat("\n")
  }

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
  .plot_marginal_means(fit, params_df, data, model_name, out_dir, is_mixed = FALSE, raw_data = df_est)

  invisible(list(fit = fit, params = params_df, metrics = metrics))
}

# ══════════════════════════════════════════════════════════════════════════════
# run_mixed_linear_panel(df_est, rhs, model_name, ...)
#   Modèle à intercepts aléatoires SANS structure temporelle. Depuis le passage
#   à nlme, ce n'est plus qu'un cas particulier de run_mixed_linear_panel_ar
#   (ar_order = 0) : même moteur, même sortie, mêmes fichiers. On délègue donc
#   plutôt que de dupliquer 200 lignes.
#
#   Deux différences avec l'ancienne implémentation lme4 :
#     • plusieurs niveaux de panel sont EMBOÎTÉS (~ 1 | a/b) et non croisés
#       ((1|a) + (1|b)). Identique ici, où les identifiants de trajet sont
#       uniques par rider, mais la formule affichée change.
#     • les LRT sur les variances (ex-`lmerTest::ranova`) sont obtenus en
#       réajustant explicitement chaque modèle nul (cf. run_mixed_linear_panel_ar).
# ══════════════════════════════════════════════════════════════════════════════
run_mixed_linear_panel <- function(df_est, rhs, model_name,
                                   panel_id_col = "rider_id",
                                   method = "ML",
                                   robust = TRUE, cluster_col = "rider_id",
                                   make_plots = TRUE, verbose = TRUE) {
  run_mixed_linear_panel_ar(df_est, rhs, model_name,
                            panel_id_col = panel_id_col,
                            ar_order     = 0,
                            method       = method,
                            make_plots   = make_plots,
                            robust       = robust,
                            cluster_col  = cluster_col,
                            verbose      = verbose)
}

# ══════════════════════════════════════════════════════════════════════════════
# run_mixed_linear_panel_ar(df_est, rhs, model_name, ...)
#   Même sortie complète que run_mixed_linear_panel (console formatée, params.csv,
#   params.tex, stats.tex, plots marginaux, $metrics) mais avec nlme::lme +
#   corARMA(p = ar_order, q = 0) : corrélation temporelle AR(p) sur les résidus
#   intra-trajet. Utilisé pour M6 (AR1) et M6b (AR2).
# ══════════════════════════════════════════════════════════════════════════════
run_mixed_linear_panel_ar <- function(df_est, rhs, model_name,
                                      panel_id_col = c("rider_id", "source"),
                                      cor_id_col   = NULL,
                                      time_col     = "second",
                                      ar_order     = 1,
                                      method       = "ML",
                                      make_plots   = TRUE,
                                      robust = TRUE, cluster_col = "rider_id",
                                      verbose = TRUE) {
  use_reml   <- identical(toupper(method), "REML")
  panel_cols <- panel_id_col
  # Groupes de la structure AR : par défaut identiques aux effets aléatoires.
  # cor_id_col permet de les découpler (ex. random ~1|rider_id mais AR blocquée
  # par trajet : panel_id_col="rider_id", cor_id_col=c("rider_id","source")).
  cor_cols   <- if (is.null(cor_id_col)) panel_cols else unique(c(panel_cols, cor_id_col))

  formula_obj <- as.formula(paste0("speed_kmh_kalman_t1 ~ ", rhs))

  vars_used <- unique(c("speed_kmh_kalman_t1", panel_cols, cor_cols,
                        if (ar_order >= 1) time_col else NULL,
                        all.vars(as.formula(paste("~", rhs)))))
  vars_used <- vars_used[vars_used %in% names(df_est)]
  data      <- df_est[, vars_used, drop = FALSE]
  for (cn in names(data)) if (is.character(data[[cn]])) data[[cn]] <- factor(data[[cn]])
  before  <- nrow(data)
  data    <- data[complete.cases(data), ]
  dropped <- before - nrow(data)
  if (dropped > 0) message(sprintf("[%s] ⚠ %d lignes supprimées (NaN)", model_name, dropped))
  .ord_cols <- data[unique(c(panel_cols, cor_cols))]
  if (ar_order >= 1) .ord_cols <- c(.ord_cols, list(data[[time_col]]))
  data <- data[do.call(order, .ord_cols), ]

  N_obs    <- nrow(data)
  N_riders <- length(unique(data[[panel_cols[1]]]))

  # ── Structures aléatoire (emboîtée) et de corrélation AR(p) ─────────────────
  nested_grp  <- paste(panel_cols, collapse = "/")
  cor_grp     <- paste(cor_cols,   collapse = "/")
  ar_start    <- switch(as.character(ar_order), "0" = NULL, "1" = 0.8,
                        "2" = c(0.5, 0.2), rep(0.2, ar_order))
  # ar_order = 0 : aucune structure de corrélation (équivalent nlme d'un simple
  # modèle à intercepts aléatoires) -> run_mixed_linear_panel délègue ici.
  .cor_struct <- function(cgrp) {
    if (ar_order < 1) return(NULL)
    nlme::corARMA(ar_start, form = as.formula(paste("~", time_col, "|", cgrp)),
                  p = ar_order, q = 0)
  }

  # ── Garde-fou : corARMA exige un temps UNIQUE dans chaque groupe de l'AR ────
  # `second` repart à 0 à chaque trajet : si les groupes AR n'incluent pas la
  # colonne trajet, les temps sont dupliqués et nlme échoue avec un message
  # peu parlant ("covariate must have unique integer values within groups").
  .dup <- ar_order >= 1 && anyDuplicated(data[, c(cor_cols, time_col), drop = FALSE]) > 0
  if (.dup) {
    stop(sprintf(paste0("[%s] `%s` n'est pas unique dans les groupes AR (~ %s | %s).\n",
                        "  corARMA exige un temps unique par groupe. Ajoutez la colonne trajet\n",
                        "  aux groupes AR sans toucher aux effets aléatoires, p. ex. :\n",
                        "    panel_id_col = c(%s), cor_id_col = c(%s, \"source\")"),
                 model_name, time_col, time_col, cor_grp,
                 paste(sprintf('"%s"', panel_cols), collapse = ", "),
                 paste(sprintf('"%s"', panel_cols), collapse = ", ")), call. = FALSE)
  }
  ctrl        <- nlme::lmeControl(opt = "nlminb", maxIter = 300, msMaxIter = 300, returnObject = TRUE)
  reml_method <- if (use_reml) "REML" else "ML"

  fit_one <- function(grp, cgrp = cor_grp) {
    nlme::lme(formula_obj,
              random      = as.formula(paste("~ 1 |", grp)),
              correlation = .cor_struct(cgrp),
              data = data, method = reml_method, control = ctrl)
  }
  used_panel <- panel_cols
  fit <- if (length(panel_cols) == 1) fit_one(nested_grp) else
    tryCatch(fit_one(nested_grp), error = function(e) {
      # Repli utile UNIQUEMENT si la structure aléatoire est emboîtée : sinon on
      # refit à l'identique le modèle qui vient d'échouer.
      message(sprintf("[%s] emboîté non convergé (%s) → repli sur ~ 1 | %s",
                      model_name, conditionMessage(e), panel_cols[length(panel_cols)]))
      used_panel <<- panel_cols[length(panel_cols)]
      fit_one(used_panel)
    })

  phi <- if (ar_order < 1) numeric(0) else
           as.numeric(coef(fit$modelStruct$corStruct, unconstrained = FALSE))

  # ── Log-vraisemblance, ρ² et LRT vs nul (même structure RE + AR, β = μ) ──────
  ll <- as.numeric(logLik(fit)); k <- attr(logLik(fit), "df")
  ll_null_ols <- as.numeric(logLik(lm(speed_kmh_kalman_t1 ~ 1, data = data)))
  fit_null <- tryCatch(
    nlme::lme(speed_kmh_kalman_t1 ~ 1,
              random      = as.formula(paste("~ 1 |", paste(used_panel, collapse = "/"))),
              correlation = .cor_struct(cor_grp),
              data = data, method = "ML", control = ctrl),
    error = function(e) NULL)
  if (!is.null(fit_null)) {
    ll_null <- as.numeric(logLik(fit_null)); k_null <- attr(logLik(fit_null), "df")
    lrt_stat <- -2 * (ll_null - ll); lrt_df <- k - k_null
    lrt_p <- if (lrt_df > 0) pchisq(lrt_stat, df = lrt_df, lower.tail = FALSE) else NA
  } else { ll_null <- NA; lrt_stat <- NA; lrt_df <- NA; lrt_p <- NA }

  rho2     <- 1 - ll / ll_null_ols
  rho2_bar <- 1 - (ll - k) / ll_null_ols
  aic_val  <- AIC(fit); bic_val <- BIC(fit)

  sig_str <- if (!is.na(lrt_p) && lrt_p < 0.001) " ***" else
             if (!is.na(lrt_p) && lrt_p < 0.01)  " **"  else
             if (!is.na(lrt_p) && lrt_p < 0.05)  " *"   else " (n.s.)"

  # ── Écarts-types aléatoires (nlme::VarCorr : outer → inner → résidu) ─────────
  sds       <- suppressWarnings(as.numeric(nlme::VarCorr(fit)[, "StdDev"]))
  sds       <- sds[!is.na(sds)]
  sigma_eps <- sds[length(sds)]
  re_sds    <- sds[-length(sds)]
  sigma_rid <- re_sds[1]
  icc       <- sigma_rid^2 / (sum(re_sds^2) + sigma_eps^2)

  # ── LRT sur les variances aléatoires (σ=0) en GARDANT l'AR ───────────────────
  #    Équivalent nlme du ranova de M5. On refit chaque modèle nul (une variance
  #    retirée) en conservant EXACTEMENT la même structure AR :
  #      H0 σ_rider=0 : random ~1|innermost , AR groupée par innermost (trajet)
  #      H0 σ_pc=0    : random sur les niveaux restants , AR sur la forme nichée
  #    Tests au BORD (variance=0) ⇒ p-value chi²(1) NAÏVE (conservatrice), comme M5.
  refit_ll <- function(rand_grp, cor_grp) tryCatch(as.numeric(logLik(
    nlme::lme(formula_obj,
              random      = as.formula(paste("~ 1 |", rand_grp)),
              correlation = .cor_struct(cor_grp),
              data = data, method = "ML", control = ctrl))),
    error = function(e) NA_real_)

  sigma_rider_lrt <- NA; sigma_rider_p <- NA
  extra_sigmas <- list(); extra_ns <- list(); extra_lrts <- list()
  if (length(used_panel) == 1) {
    # H0 : sigma_rider = 0 avec un seul niveau aléatoire -> gls (mêmes AR + betas)
    ll_gls <- tryCatch(as.numeric(logLik(
      nlme::gls(formula_obj,
                correlation = .cor_struct(cor_grp),
                data = data, method = "ML",
                control = nlme::glsControl(maxIter = 300, msMaxIter = 300, returnObject = TRUE)))),
      error = function(e) NA_real_)
    if (!is.na(ll_gls)) {
      sigma_rider_lrt <- round(max(0, -2 * (ll_gls - ll)), 2)
      sigma_rider_p   <- pchisq(sigma_rider_lrt, df = 1, lower.tail = FALSE)
    }
  }
  if (length(used_panel) > 1) {
    innermost <- used_panel[length(used_panel)]
    full_grp  <- paste(used_panel, collapse = "/")
    # H0 : σ_rider = 0 (retirer le 1er niveau ; AR groupée par l'innermost)
    ll_no_rider <- refit_ll(innermost, innermost)
    if (!is.na(ll_no_rider)) {
      sigma_rider_lrt <- round(max(0, -2 * (ll_no_rider - ll)), 2)
      sigma_rider_p   <- pchisq(sigma_rider_lrt, df = 1, lower.tail = FALSE)
    }
    # H0 : σ_pc = 0 pour chaque niveau supplémentaire (AR nichée conservée)
    for (j in seq_along(used_panel[-1])) {
      pc <- used_panel[j + 1]
      extra_sigmas[[pc]] <- if (length(re_sds) >= j + 1) round(re_sds[j + 1], 4) else NA
      extra_ns[[pc]]     <- length(unique(data[[pc]]))
      grp_keep <- paste(setdiff(used_panel, pc), collapse = "/")
      ll_no_pc <- refit_ll(grp_keep, full_grp)
      if (!is.na(ll_no_pc)) {
        st <- round(max(0, -2 * (ll_no_pc - ll)), 2)
        extra_lrts[[pc]] <- list(lrt = st, p = pchisq(st, df = 1, lower.tail = FALSE))
      } else {
        extra_lrts[[pc]] <- list(lrt = NA, p = NA)
      }
    }
  }

  # ── R² marginal / conditionnel (Nakagawa) ───────────────────────────────────
  r2 <- tryCatch(MuMIn::r.squaredGLMM(fit), error = function(e)
          matrix(NA, 1, 2, dimnames = list(NULL, c("R2m", "R2c"))))
  r2_marginal <- r2[1, "R2m"]; r2_conditional <- r2[1, "R2c"]

  # ── Table des paramètres au format lmer (pour .params_to_latex / plots) ──────
  tt <- summary(fit)$tTable
  params_df <- data.frame(
    Estimate     = tt[, "Value"],
    `Std. Error` = tt[, "Std.Error"],
    df           = tt[, "DF"],
    `t value`    = tt[, "t-value"],
    `Pr(>|t|)`   = tt[, "p-value"],
    check.names  = FALSE, row.names = rownames(tt)
  )

  if (isTRUE(verbose)) {
  cat(sprintf("\n%s\n", strrep("=", 72)))
  cat(sprintf("  Mixed panel linear model%s: %s  [method: %s]\n",
              if (ar_order < 1) "" else sprintf(" + AR(%d)", ar_order),
              model_name, toupper(method)))
  cat(sprintf("  Panel: %s%s\n", paste(used_panel, collapse = " + "),
              if (ar_order < 1) "" else sprintf("  |  temps: %s", time_col)))
  cat(sprintf("  Riders=%d  Observations=%d\n", N_riders, N_obs))
  cat(sprintf("  sigma_%s=%.4f  sigma_eps=%.4f\n", used_panel[1], sigma_rid, sigma_eps))
  if (ar_order >= 1)
    cat(sprintf("  phi (AR%d) = %s\n", ar_order, paste(sprintf("%.4f", phi), collapse = ", ")))
  .st_trip <- if (!is.null(extra_lrts[["source"]])) extra_lrts[["source"]]$lrt else NA
  .sp_trip <- if (!is.null(extra_lrts[["source"]])) extra_lrts[["source"]]$p   else NA
  cat(sprintf("  LRT sigma_rider=0 : chi2(1)=%s (p=%s) | LRT sigma_trip=0 : chi2(1)=%s (p=%s)\n",
              ifelse(is.na(sigma_rider_lrt), "---", sprintf("%.2f", sigma_rider_lrt)),
              ifelse(is.na(sigma_rider_p),   "---", sprintf("%.4f", sigma_rider_p)),
              ifelse(is.na(.st_trip),        "---", sprintf("%.2f", .st_trip)),
              ifelse(is.na(.sp_trip),        "---", sprintf("%.4f", .sp_trip))))
  cat(sprintf("  Modèle nul      : LL=%s\n", if (is.na(ll_null)) "NA" else sprintf("%.2f", ll_null)))
  cat(sprintf("  Modèle principal: K=%d  LL=%.2f\n", k, ll))
  cat(sprintf("  Rm²=%.4f  Rc²=%.4f\n", r2_marginal, r2_conditional))
  cat(sprintf("  rho²=%.4f  AIC=%.1f  BIC=%.1f\n", rho2, aic_val, bic_val))
  cat(sprintf("  LRT vs nul : chi²(%s)=%s  p=%s%s\n",
              ifelse(is.na(lrt_df), "NA", lrt_df),
              ifelse(is.na(lrt_stat), "NA", sprintf("%.2f", lrt_stat)),
              ifelse(is.na(lrt_p), "NA", sprintf("%.4f", lrt_p)), sig_str))
  cat(sprintf("%s\n", strrep("=", 72)))
  print(params_df)

  # ── Corrélations entre betas estimés ────────────────────────────────────────
  cor_beta <- cov2cor(as.matrix(vcov(fit)))
  cor_beta[lower.tri(cor_beta, diag = TRUE)] <- NA
  idx_b <- which(!is.na(cor_beta), arr.ind = TRUE)
  if (nrow(idx_b) > 0) {
    cbp <- data.frame(var1 = rownames(cor_beta)[idx_b[, 1]],
                      var2 = colnames(cor_beta)[idx_b[, 2]],
                      r    = cor_beta[idx_b], stringsAsFactors = FALSE)
    cbp <- cbp[order(abs(cbp$r), decreasing = TRUE), ]
    cat(sprintf("\n  [%s] Corrélations entre betas estimés (|r| décroissant) :\n", model_name))
    for (i in seq_len(min(nrow(cbp), 15))) {
      flag <- if (abs(cbp$r[i]) > 0.7) "  ⚠ > 0.7" else ""
      cat(sprintf("    cor(%-28s, %-28s) = %+.3f%s\n", cbp$var1[i], cbp$var2[i], cbp$r[i], flag))
    }
    cat("\n")
  }
  }   # fin du bloc console (verbose)

  metrics <- list(
    Model = model_name, N = N_obs, N_riders = N_riders, K = k,
    LL_null = round(ll_null_ols, 2), LL_final = round(ll, 2),
    rho2 = round(rho2, 4), rho2_bar = round(rho2_bar, 4),
    AIC = round(aic_val, 2), BIC = round(bic_val, 2),
    LRT_stat = if (!is.na(lrt_stat)) round(lrt_stat, 2) else NA, LRT_df = lrt_df,
    LRT_p = if (!is.na(lrt_p)) round(lrt_p, 4) else NA,
    r2_marginal = round(r2_marginal, 4), r2_conditional = round(r2_conditional, 4),
    sigma_rider = round(sigma_rid, 4), sigma_eps = round(sigma_eps, 4), ICC = round(icc, 4),
    sigma_rider_lrt = sigma_rider_lrt,
    sigma_rider_p   = if (!is.na(sigma_rider_p)) round(sigma_rider_p, 4) else NA,
    extra_sigmas = extra_sigmas, extra_ns = extra_ns, extra_lrts = extra_lrts,
    phi = round(phi, 4), ar_order = ar_order
  )

  # ── Sauvegardes (params.csv, params.tex, stats.tex, plots) ──────────────────
  out_dir <- .get_out_dir(model_name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  robust_df <- if (isTRUE(robust)) .robust_params_df(fit, cluster_col) else NULL
  write.csv(.merge_robust_csv(params_df, robust_df),
            file.path(out_dir, paste0(model_name, "_params.csv")))

  sigmas_list <- list(list(label = "$\\sigma_{\\text{rider}}$", value = metrics$sigma_rider))
  if (length(metrics$extra_sigmas) > 0) {
    for (pc in names(metrics$extra_sigmas)) {
      lbl <- if (pc == "source") "$\\sigma_{\\text{trip}}$" else sprintf("$\\sigma_{\\text{%s}}$", pc)
      sigmas_list <- c(sigmas_list, list(list(label = lbl, value = metrics$extra_sigmas[[pc]])))
    }
  }
  sigmas_list <- c(sigmas_list, list(list(label = "$\\sigma_{\\varepsilon}$", value = metrics$sigma_eps)))

  writeLines(.params_to_latex(params_df, model_name,
               equation = .build_equation_latex(params_df, mixed = TRUE, panel_cols = used_panel),
               sigmas   = sigmas_list, robust_df = robust_df),
             file.path(out_dir, paste0(model_name, "_params.tex")))
  writeLines(.stats_to_latex(metrics, model_name),
             file.path(out_dir, paste0(model_name, "_stats.tex")))
  if (make_plots) {
    tryCatch(.plot_marginal_means(fit, params_df, data, model_name, out_dir,
                                  is_mixed = TRUE, raw_data = df_est),
             error = function(e) message(sprintf("[%s] plots marginaux ignorés : %s",
                                                 model_name, conditionMessage(e))))
  }

  invisible(list(fit = fit, params = params_df, metrics = metrics))
}

# ══════════════════════════════════════════════════════════════════════════════
# fit_mixed_nlme(df_est, rhs, panel_id_col, ...)
#   Ajuste un modèle à intercepts aléatoires avec nlme::lme (SANS structure AR).
#   Même modèle que run_mixed_linear_panel (lme4::lmer) mais sur le moteur nlme,
#   pour que M5 et M6 soient comparables sans effet de changement de paquet.
# ══════════════════════════════════════════════════════════════════════════════
fit_mixed_nlme <- function(df_est, rhs, panel_id_col = "rider_id",
                           method = "ML", target = "speed_kmh_kalman_t1") {
  f  <- as.formula(paste(target, "~", rhs))
  vs <- unique(c(target, panel_id_col, all.vars(as.formula(paste("~", rhs)))))
  d  <- df_est[, vs[vs %in% names(df_est)], drop = FALSE]
  for (cn in names(d)) if (is.character(d[[cn]])) d[[cn]] <- factor(d[[cn]])
  before <- nrow(d); d <- d[complete.cases(d), ]
  if (before > nrow(d)) message(sprintf("[fit_mixed_nlme] ⚠ %d lignes supprimées (NaN)",
                                        before - nrow(d)))
  nlme::lme(f,
            random  = as.formula(paste("~ 1 |", paste(panel_id_col, collapse = "/"))),
            data    = d,
            method  = if (identical(toupper(method), "REML")) "REML" else "ML",
            control = nlme::lmeControl(opt = "nlminb", maxIter = 300,
                                       msMaxIter = 300, returnObject = TRUE))
}

# ══════════════════════════════════════════════════════════════════════════════
# print_robust_tp(fit, cluster_col, model_name)
#   Affiche UNIQUEMENT le t et la p-value cluster-robustes CR2 (+ étoiles).
#   S'appuie sur .robust_params_df (CR2 + dof de Satterthwaite) : aucune
#   estimation n'est refaite, seules les colonnes affichées sont restreintes.
#   Marche sur lme4::lmer comme sur nlme::lme.
# ══════════════════════════════════════════════════════════════════════════════
print_robust_tp <- function(fit, cluster_col = "rider_id", model_name = NULL,
                            digits = 3, show_df = FALSE) {
  pf <- .robust_params_df(fit, cluster_col)
  if (is.null(pf)) {
    message("[print_robust_tp] colonnes robustes indisponibles (clubSandwich ?)")
    return(invisible(NULL))
  }
  st <- function(p) ifelse(is.na(p), "", ifelse(p < .001, "***", ifelse(p < .01, "**",
                    ifelse(p < .05, "*", ifelse(p < .1, ".", "")))))
  out <- data.frame(Terme = rownames(pf), check.names = FALSE,
                    stringsAsFactors = FALSE)
  if (isTRUE(show_df)) out[["df"]] <- round(pf[["df"]], 2)
  out[["t robuste"]] <- round(pf[["t value"]], digits)
  out[["p robuste"]] <- signif(pf[["Pr(>|t|)"]], digits)
  out[[" "]]         <- st(pf[["Pr(>|t|)"]])
  cat(sprintf("%s— t et p cluster-robustes CR2 (cluster = %s, %s clusters)\n\n",
              if (is.null(model_name)) "" else paste0(model_name, " "),
              cluster_col, attr(pf, "n_clusters")))
  print(out, row.names = FALSE)
  invisible(out)
}

cat("✔ Fonctions R chargées : run_linear, run_mixed_linear_panel, run_mixed_linear_panel_ar,\n  fit_mixed_nlme, print_robust_tp\n")
