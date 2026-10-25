# SPDX-License-Identifier: GPL-2.0-or-later
# Optional fixed complete-case binomial and Poisson comparisons from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_glm_missing_r.R LIBRARY OUTPUT_JSON")
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

base <- expand.grid(x1 = 0:3, x2 = 0:3, rep = 1:8,
                    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
base$z <- as.numeric(base$rep %% 3)
base$g <- factor(c("A", "B", "C")[1L + (base$x1 + 2L*base$x2 + base$rep) %% 3L],
                 levels = c("A", "B", "C"))
binary <- base
binary$y <- as.integer((2L*base$x1 + 3L*base$x2 + 5L*base$rep) %% 11L <
                       (4L + base$x1))
count <- base
count$y <- as.integer((2L*base$x1 + 3L*base$x2 + 5L*base$rep) %% 9L)
new <- data.frame(x1 = c(12, 64, 120)/128, x2 = c(1, 2, 3),
                  z = c(0, 1, 2),
                  g = factor(c("A", "B", "C"), levels = levels(base$g)))

make_case <- function(family_name, mode) {
  d <- if (family_name == "binomial") binary else count
  q <- if (mode == "quantized") 4L else NULL
  predict_rows <- new
  if (mode == "quantized") {
    d$x1 <- seq_len(nrow(d))/nrow(d)
    d$y[1:20] <- NA
    d$x1[21] <- NA
    d$z[22] <- NA
    d$g[23] <- NA
  } else {
    predict_rows$x1 <- c(0, 1, 3)
    d$y[1] <- NA
    d$x1[2] <- NA
    d$z[3] <- NA
    d$g[4] <- NA
  }
  model <- qgcomp::qgcomp.glm.noboot(
      y ~ x1 + x2 + z + g, data = d, expnms = c("x1", "x2"), q = q,
      family = if (family_name == "binomial") stats::binomial() else stats::poisson())
  used <- as.integer(rownames(model$fit$model))
  scored <- if (is.null(q)) d else qgcomp::quantize(d, c("x1", "x2"), q = q)$data
  new_scored <- if (is.null(q)) predict_rows else
      qgcomp::quantize(predict_rows, c("x1", "x2"), q = q,
                        breaks = model$breaks)$data
  list(
    name = paste(family_name, mode, sep = "_"),
    inputs = list(
      family = family_name, formula = "y ~ x1 + x2 + z + C(g)",
      exposures = c("x1", "x2"), q = q, missing = "drop",
      data = list(y = unname(d$y), x1 = unname(d$x1), x2 = unname(d$x2),
                  z = unname(d$z), g = unname(as.character(d$g))),
      new_data = list(x1 = unname(predict_rows$x1), x2 = unname(predict_rows$x2),
                      z = unname(predict_rows$z),
                      g = unname(as.character(predict_rows$g)))
    ),
    expected = list(
      original_row_count = nrow(d), training_row_indices = used - 1L,
      coefficient_names = names(stats::coef(model$fit)),
      coefficients = unname(stats::coef(model$fit)),
      covariance = unname(summary(model$fit)$cov.scaled),
      scores = unname(as.matrix(scored[used, c("x1", "x2")])),
      breaks = model$breaks, psi = unname(model$psi),
      variance = unname(model$var.psi), se = unname(sqrt(model$var.psi)),
      ci = unname(model$ci), zstat = unname(tail(model$zstat, 1L)),
      p = unname(tail(model$pval, 1L)), residual_df = model$fit$df.residual,
      fitted = unname(stats::fitted(model$fit)),
      predictions = unname(stats::predict(model$fit, newdata = new_scored,
                                           type = "response"))
    )
  )
}

cases <- list(make_case("binomial", "quantized"),
              make_case("binomial", "prequantized"),
              make_case("poisson", "quantized"),
              make_case("poisson", "prequantized"))
payload <- list(R_version = R.version.string,
                qgcomp_version = as.character(utils::packageVersion("qgcomp")),
                jsonlite_version = as.character(utils::packageVersion("jsonlite")),
                qgcomp_origin = normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
                library = lib, cases = cases)
jsonlite::write_json(payload, out, pretty = TRUE, auto_unbox = TRUE,
                     digits = NA, null = "null", na = "null")
cat(length(cases), "installed-R complete-case GLM cases exported\n")
