# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed complete-case Gaussian comparison from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_missing_r.R LIBRARY OUTPUT_JSON")
lib <- normalizePath(args[[1L]], mustWork = TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) {
  stop("OUTPUT_JSON already exists")
}
.libPaths(c(lib, normalizePath(.Library, mustWork = TRUE)), include.site = FALSE)
if (as.character(utils::packageVersion("qgcomp", lib.loc = lib)) != "2.19.6") {
  stop("installed qgcomp 2.19.6 is required")
}
if (!requireNamespace("qgcomp", lib.loc = lib, quietly = TRUE) ||
    !requireNamespace("jsonlite", lib.loc = lib, quietly = TRUE)) {
  stop("qgcomp/jsonlite or dependencies are missing")
}
if (!identical(normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
               normalizePath(file.path(lib, "qgcomp")))) {
  stop("loaded qgcomp namespace is outside the requested library")
}

base <- expand.grid(x1 = 0:3, x2 = 0:3, g = c("A", "B", "C"),
                    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
base$g <- factor(base$g, levels = c("A", "B", "C"))
base$y <- 2 + 1.5 * base$x1 - .4 * base$x2 + .7 * (base$g == "B") -
          .3 * (base$g == "C") + .2 * (base$x1 - 1.5) * (base$x2 - 1.5)
base$z <- seq_len(nrow(base)) %% 7
new <- data.frame(x1 = c(1, 2), x2 = c(2, 3), z = c(4, 5),
                  g = factor(c("B", "C"), levels = levels(base$g)))

make_case <- function(name, d, formula, case_weights = NULL) {
  if (is.null(case_weights)) {
    model <- qgcomp::qgcomp.glm.noboot(formula, data = d,
        expnms = c("x1", "x2"), q = 4L, family = stats::gaussian())
  } else {
    w <- case_weights
    model <- qgcomp::qgcomp.glm.noboot(formula, data = d,
        expnms = c("x1", "x2"), q = 4L, weights = w,
        family = stats::gaussian())
  }
  mod <- summary(model$fit)
  used <- as.integer(rownames(model$fit$model))
  scored <- qgcomp::quantize(d, c("x1", "x2"), q = 4L)$data
  new_scored <- qgcomp::quantize(new, c("x1", "x2"),
                                  q = 4L, breaks = model$breaks)$data
  list(
    name = name,
    inputs = list(
      formula = "y ~ x1 + x2 + z + C(g)", exposures = c("x1", "x2"),
      data = list(y = unname(d$y), x1 = unname(d$x1), x2 = unname(d$x2),
                  z = unname(d$z), g = unname(as.character(d$g))),
      new_data = list(x1 = unname(new$x1), x2 = unname(new$x2),
                      z = unname(new$z), g = unname(as.character(new$g))),
      q = 4L, weights = if (is.null(case_weights)) NULL else unname(case_weights),
      missing = "drop"
    ),
    expected = list(
      original_row_count = nrow(d),
      training_row_indices = used - 1L,
      coefficient_names = names(stats::coef(model$fit)),
      coefficients = unname(stats::coef(model$fit)),
      covariance = unname(mod$cov.scaled),
      scores = unname(as.matrix(scored[used, c("x1", "x2")])),
      breaks = model$breaks,
      psi = unname(model$psi), variance = unname(model$var.psi),
      se = unname(sqrt(model$var.psi)), ci = unname(model$ci),
      t = unname(tail(model$tstat, 1L)), p = unname(tail(model$pval, 1L)),
      p_df = model$df, residual_df = model$fit$df.residual,
      dispersion = mod$dispersion,
      fitted = unname(stats::fitted(model$fit)),
      predictions = unname(stats::predict(model$fit, newdata = new_scored,
                                           type = "response"))
    )
  )
}

d1 <- base
d1$y[1] <- NA_real_
d1$z[2] <- NA_real_
d1$x1[3] <- NA_real_
d2 <- base
d2$g[1] <- NA
w <- .5 + (seq_len(nrow(d2)) %% 7) / 4
w[2] <- NA_real_
cases <- list(make_case("missing_outcome_covariate_exposure", d1,
                        y ~ x1 + x2 + z + g),
              make_case("missing_category_and_weight", d2,
                        y ~ x1 + x2 + z + g, w))
payload <- list(
  R_version = R.version.string,
  qgcomp_version = as.character(utils::packageVersion("qgcomp")),
  jsonlite_version = as.character(utils::packageVersion("jsonlite")),
  qgcomp_origin = normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
  library = lib, cases = cases
)
jsonlite::write_json(payload, out, pretty = TRUE, auto_unbox = TRUE,
                     digits = NA, null = "null", na = "null")
cat(length(cases), "installed-R missing-row cases exported\n")
