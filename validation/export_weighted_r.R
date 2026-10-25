# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed positive-case-weight comparisons from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_weighted_r.R LIBRARY OUTPUT_JSON")
lib <- normalizePath(args[[1L]], mustWork = TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) {
  stop("OUTPUT_JSON already exists")
}
base_lib <- normalizePath(.Library, mustWork = TRUE)
.libPaths(c(lib, base_lib), include.site = FALSE)
if (!setequal(.libPaths(), unique(c(lib, base_lib)))) stop("R library isolation failed")
if (as.character(utils::packageVersion("qgcomp", lib.loc = lib)) != "2.19.6") {
  stop("installed qgcomp 2.19.6 is required")
}
if (!requireNamespace("jsonlite", lib.loc = lib, quietly = TRUE) ||
    !requireNamespace("qgcomp", lib.loc = lib, quietly = TRUE)) {
  stop("qgcomp/jsonlite or mandatory dependencies are missing")
}
if (!identical(normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
               normalizePath(file.path(lib, "qgcomp")))) {
  stop("loaded qgcomp namespace is outside the requested library")
}

make_case <- function(name, y, X, Z = NULL, weights, q = 4L, new_X = NULL,
                      new_Z = NULL) {
  X <- as.matrix(X)
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  d <- as.data.frame(X)
  new_d <- as.data.frame(if (is.null(new_X)) X[seq_len(min(5L, nrow(X))), , drop = FALSE]
                         else new_X)
  colnames(new_d) <- colnames(X)
  cov_names <- character(0)
  if (!is.null(Z)) {
    Z <- as.matrix(Z)
    colnames(Z) <- paste0("z", seq_len(ncol(Z)))
    cov_names <- colnames(Z)
    d <- cbind(d, as.data.frame(Z))
    if (is.null(new_Z)) new_Z <- Z[seq_len(nrow(new_d)), , drop = FALSE]
    new_Z <- as.matrix(new_Z)
    colnames(new_Z) <- cov_names
    new_d <- cbind(new_d, as.data.frame(new_Z))
  }
  d$y <- as.numeric(y)
  f <- stats::reformulate(c(colnames(X), cov_names), response = "y")
  model <- qgcomp::qgcomp.glm.noboot(f, data = d, expnms = colnames(X), q = q,
                                      weights = weights, family = stats::gaussian())
  summary_model <- summary(model$fit)
  quantized <- if (is.null(q)) d else qgcomp::quantize(d, expnms = colnames(X), q = q)$data
  new_quantized <- if (is.null(q)) new_d else {
    qgcomp::quantize(new_d, expnms = colnames(X), q = q,
                      breaks = model$breaks)$data
  }
  list(
    name = name,
    inputs = list(y = unname(as.numeric(y)), exposures = unname(X),
                  covariates = if (is.null(Z)) NULL else unname(Z),
                  weights = unname(as.numeric(weights)), q = q,
                  new_exposures = unname(as.matrix(new_d[, colnames(X), drop = FALSE])),
                  new_covariates = if (is.null(Z)) NULL else
                    unname(as.matrix(new_d[, cov_names, drop = FALSE]))),
    expected = list(coefficients = unname(stats::coef(model$fit)),
                    covariance = unname(summary_model$cov.scaled),
                    scores = unname(as.matrix(quantized[, colnames(X), drop = FALSE])),
                    breaks = model$breaks, psi = unname(model$psi),
                    variance = unname(model$var.psi),
                    se = unname(sqrt(model$var.psi)), ci = unname(model$ci),
                    t = unname(tail(model$tstat, 1L)),
                    p = unname(tail(model$pval, 1L)),
                    p_df = model$df, residual_df = model$fit$df.residual,
                    dispersion = summary_model$dispersion,
                    fitted = unname(stats::fitted(model$fit)),
                    predictions = unname(stats::predict(model$fit,
                      newdata = new_quantized, type = "response")),
                    positive_psi = model$pos.psi,
                    negative_psi = model$neg.psi,
                    positive_weights = as.list(model$pos.weights),
                    negative_weights = as.list(model$neg.weights)))
}

i <- 0:63
X <- cbind(i %% 8 + i / 1000, (3 * i) %% 11 + .3 * (i %% 8))
Z <- matrix(cos(i * .4), ncol = 1L)
y <- 1.25 + .6 * X[, 1L] - .5 * X[, 2L] + .2 * Z[, 1L] + sin((i + 1L) * 1.7) / 2
w <- .5 + (i %% 7) / 4
X2 <- cbind(c(0,0,0,0,1,1,1,1), c(0,0,1,1,0,0,1,1))
y2 <- c(1,2,3,4,2,2,5,6)
w2 <- 1:8
cases <- list(make_case("weighted_raw_adjusted", y, X, Z, w),
              make_case("weighted_prequantized_correlated", y2, X2,
                        weights = w2, q = NULL))
metadata <- list(reference = "installed R qgcomp 2.19.6",
                 R_version = R.version.string,
                 qgcomp_version = as.character(utils::packageVersion("qgcomp")),
                 jsonlite_version = as.character(utils::packageVersion("jsonlite")),
                 qgcomp_origin = normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
                 library = lib)
payload <- c(metadata, list(cases = cases))
jsonlite::write_json(payload, out, pretty = TRUE, auto_unbox = TRUE,
                     digits = NA, null = "null")
cat(length(cases), "installed-R weighted cases exported\n")
