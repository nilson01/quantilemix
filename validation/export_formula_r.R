# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed formula/categorical comparison from isolated installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_formula_r.R LIBRARY OUTPUT_JSON")
lib <- normalizePath(args[[1L]], mustWork = TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop("OUTPUT_JSON already exists")
.libPaths(c(lib, normalizePath(.Library, mustWork = TRUE)), include.site = FALSE)
if (as.character(utils::packageVersion("qgcomp", lib.loc = lib)) != "2.19.6") {
  stop("installed qgcomp 2.19.6 is required")
}
if (!requireNamespace("qgcomp", lib.loc = lib, quietly = TRUE) ||
    !requireNamespace("jsonlite", lib.loc = lib, quietly = TRUE)) {
  stop("qgcomp/jsonlite or their dependencies are missing")
}
if (!identical(normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
               normalizePath(file.path(lib, "qgcomp")))) {
  stop("loaded qgcomp namespace is outside the requested library")
}

d <- expand.grid(x1 = 0:3, x2 = 0:3, g = c("A", "B", "C"),
                 KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
d$g <- factor(d$g, levels = c("A", "B", "C"))
d$y <- 2 + 1.5 * d$x1 - .4 * d$x2 + .7 * (d$g == "B") -
       .3 * (d$g == "C") + .2 * (d$x1 - 1.5) * (d$x2 - 1.5)
new <- data.frame(x1 = c(1, 2), x2 = c(2, 3),
                  g = factor(c("B", "C"), levels = levels(d$g)))

make_case <- function(name, q, weights = NULL) {
  f <- y ~ x1 + x2 + g
  if (is.null(weights)) {
    model <- qgcomp::qgcomp.glm.noboot(f, data = d, expnms = c("x1", "x2"),
                                        q = q, family = stats::gaussian())
  } else {
    model <- qgcomp::qgcomp.glm.noboot(f, data = d, expnms = c("x1", "x2"),
                                        q = q, weights = weights,
                                        family = stats::gaussian())
  }
  mod <- summary(model$fit)
  scored <- if (is.null(q)) d else qgcomp::quantize(d, c("x1", "x2"), q = q)$data
  new_scored <- if (is.null(q)) new else
    qgcomp::quantize(new, c("x1", "x2"), q = q, breaks = model$breaks)$data
  list(
    name = name,
    inputs = list(
      formula = "y ~ x1 + x2 + C(g)",
      exposures = c("x1", "x2"),
      data = list(y = unname(d$y), x1 = unname(d$x1), x2 = unname(d$x2),
                  g = unname(as.character(d$g))),
      new_data = list(x1 = unname(new$x1), x2 = unname(new$x2),
                      g = unname(as.character(new$g))),
      q = q, weights = if (is.null(weights)) NULL else unname(weights)
    ),
    expected = list(
      coefficient_names = names(stats::coef(model$fit)),
      coefficients = unname(stats::coef(model$fit)),
      covariance = unname(mod$cov.scaled),
      scores = unname(as.matrix(scored[, c("x1", "x2")])),
      breaks = model$breaks,
      psi = unname(model$psi),
      variance = unname(model$var.psi),
      se = unname(sqrt(model$var.psi)),
      ci = unname(model$ci),
      t = unname(tail(model$tstat, 1L)),
      p = unname(tail(model$pval, 1L)),
      p_df = model$df,
      residual_df = model$fit$df.residual,
      dispersion = mod$dispersion,
      fitted = unname(stats::fitted(model$fit)),
      predictions = unname(stats::predict(model$fit, newdata = new_scored,
                                           type = "response"))
    )
  )
}

w <- .5 + (seq_len(nrow(d)) %% 7) / 4
cases <- list(make_case("formula_categorical_quantized", 4L),
              make_case("formula_categorical_prequantized_weighted", NULL, w))
payload <- list(
  R_version = R.version.string,
  qgcomp_version = as.character(utils::packageVersion("qgcomp")),
  jsonlite_version = as.character(utils::packageVersion("jsonlite")),
  qgcomp_origin = normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
  library = lib,
  cases = cases
)
jsonlite::write_json(payload, out, pretty = TRUE, auto_unbox = TRUE,
                     digits = NA, null = "null")
cat(length(cases), "installed-R formula cases exported\n")
