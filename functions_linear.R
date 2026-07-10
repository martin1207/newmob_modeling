# ══════════════════════════════════════════════════════════════════════════════
# functions_linear.R
# Equivalent R de funtion_linear.py
# Packages requis : lme4, lmerTest
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)   # ajoute les p-valeurs aux lmer (approx. Satterthwaite)
  if (!requireNamespace("MuMIn",     quietly = TRUE)) install.packages("MuMIn",     repos = "https://cloud.r-project.org")
  if (!requireNamespace("showtext",  quietly = TRUE)) install.packages("showtext",  repos = "https://cloud.r-project.org")
  if (!requireNamespace("sysfonts",  quietly = TRUE)) install.packages("sysfonts",  repos = "https://cloud.r-project.org")
  if (!requireNamespace("ggeffects", quietly = TRUE)) install.packages("ggeffects", repos = "https://cloud.r-project.org")
  library(MuMIn)      # r.squaredGLMM : R² marginal et conditionnel
  library(nlme)       # lme + corAR1 : modèle AR(1) sur les résidus
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
          pred_vals <- predict(fit, newdata = grid, re.form = NA)
        } else {
          pred_vals <- predict(fit, newdata = grid)
        }

        # IC via méthode delta (effets fixes uniquement)
        X       <- model.matrix(formula(fit, fixed.only = TRUE), data = grid)
        vc_fix  <- as.matrix(vcov(fit))
        common  <- intersect(colnames(X), colnames(vc_fix))
        X_sub   <- X[, common, drop = FALSE]
        vc_sub  <- vc_fix[common, common, drop = FALSE]
        var_ci  <- pmax(0, diag(X_sub %*% vc_sub %*% t(X_sub)))
        ci_lo   <- pred_vals - 1.96 * sqrt(var_ci)
        ci_hi   <- pred_vals + 1.96 * sqrt(var_ci)

        # IP pour une nouvelle observation d'un nouveau groupe (rider + trip) :
        #   se_PI = sqrt(var_IC_fixes + σ²_ε + σ²_rider + σ²_trip + ...)
        # Tous les composants aléatoires sont extraits via lme4::VarCorr().
        if (is_mixed) {
          var_resid <- sigma(fit)^2
          var_ranef <- sum(sapply(lme4::VarCorr(fit),
                                  function(vc) {
                                    v <- diag(as.matrix(vc))
                                    sum(v[is.finite(v) & v > 0])
                                  }))
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

  formula_obj  <- as.formula(paste0("speed_kmh_kalman_t1 ~ ", rhs, " + ", re_terms))
  formula_null <- as.formula(paste0("speed_kmh_kalman_t1 ~ ", re_terms))

  vars_used <- unique(c("speed_kmh_kalman_t1", panel_cols,
                         all.vars(as.formula(paste("~", rhs)))))
  vars_used <- vars_used[vars_used %in% names(df_est)]
  data      <- df_est[, vars_used, drop = FALSE]
  # Convertir les colonnes character en factor pour que lmer encode correctement
  for (cn in names(data)) {
    if (is.character(data[[cn]])) data[[cn]] <- factor(data[[cn]])
  }
  before    <- nrow(data)
  data      <- data[complete.cases(data), ]
  dropped   <- before - nrow(data)
  if (dropped > 0) message(sprintf("[%s] ⚠ %d lignes supprimées (NaN)", model_name, dropped))

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
  ll_null_ols <- as.numeric(logLik(lm(speed_kmh_kalman_t1 ~ 1, data = data)))
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
  .plot_marginal_means(fit, params_df, data, model_name, out_dir, is_mixed = TRUE, raw_data = df_est)

  invisible(list(fit = fit, params = params_df, metrics = metrics))
}

cat("✔ Fonctions R chargées : run_linear, run_mixed_linear_panel\n")
