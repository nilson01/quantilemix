# SPDX-License-Identifier: GPL-2.0-or-later
# Export the installed R package's results on the existing fixed inputs.
# This script neither installs packages nor changes the source fixtures.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: Rscript --vanilla validation/export_installed_r.R LIBRARY FIXTURES_JSON OUTPUT_JSON")
}
isolated_library <- normalizePath(args[[1]], mustWork = TRUE)
fixture_file <- normalizePath(args[[2]], mustWork = TRUE)
output_file <- args[[3]]
if (!dir.exists(isolated_library)) stop("LIBRARY must be an existing directory")
if (identical(normalizePath(output_file, mustWork = FALSE), fixture_file)) {
  stop("OUTPUT_JSON must not overwrite FIXTURES_JSON")
}
output_link <- Sys.readlink(output_file)
if (file.exists(output_file) || (!is.na(output_link) && nzchar(output_link))) {
  stop("OUTPUT_JSON already exists; choose a new output path")
}

# include.site=FALSE excludes both site libraries and user-library fallback.
# Only base/recommended packages may be loaded from .Library; audit below.
base_library <- normalizePath(.Library, mustWork = TRUE)
.libPaths(c(isolated_library, base_library), include.site = FALSE)
if (!setequal(.libPaths(), unique(c(isolated_library, base_library)))) {
  stop("Could not establish the requested isolated library search path")
}
qgcomp_location <- find.package("qgcomp", lib.loc = isolated_library, quiet = TRUE)
if (length(qgcomp_location) != 1L ||
    !identical(normalizePath(qgcomp_location),
               normalizePath(file.path(isolated_library, "qgcomp"), mustWork = FALSE))) {
  stop("qgcomp must be installed in the supplied library")
}
if (as.character(utils::packageVersion("qgcomp", lib.loc = isolated_library)) != "2.19.6") {
  stop("The supplied library must contain qgcomp 2.19.6")
}
if (!requireNamespace("jsonlite", lib.loc = isolated_library, quietly = TRUE)) {
  stop("jsonlite is required in the supplied library; no packages were installed")
}
if (!requireNamespace("qgcomp", lib.loc = isolated_library, quietly = TRUE)) {
  stop("qgcomp or its dependencies could not be loaded from the allowed libraries")
}
qgcomp_library_verified <- identical(
  normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
  normalizePath(qgcomp_location)
)
if (!qgcomp_library_verified || as.character(getNamespaceVersion("qgcomp")) != "2.19.6") {
  stop("The loaded qgcomp namespace does not match the requested installed package")
}

namespace_metadata <- function() {
  packages <- sort(loadedNamespaces())
  origins <- vapply(packages, function(package) {
    if (identical(package, "base")) return("base-recommended")
    location <- normalizePath(getNamespaceInfo(asNamespace(package), "path"))
    library <- dirname(location)
    if (identical(library, isolated_library)) return("isolated")
    priority <- read.dcf(file.path(location, "DESCRIPTION"), fields = "Priority")[1L, 1L]
    if (identical(library, base_library) && !is.na(priority) &&
        priority %in% c("base", "recommended")) return("base-recommended")
    "outside-allowed-libraries"
  }, character(1))
  unexpected <- names(origins)[origins == "outside-allowed-libraries"]
  if (length(unexpected)) {
    stop("Loaded namespaces outside the isolated/base/recommended boundary: ",
         paste(unexpected, collapse = ", "))
  }
  list(
    package_versions = as.list(setNames(vapply(packages, function(package) {
      if (identical(package, "base")) as.character(getRversion()) else {
        as.character(getNamespaceVersion(package))
      }
    }, character(1)), packages)),
    package_library_origins = as.list(origins),
    unexpected_loaded_namespaces = unname(unexpected)
  )
}
invisible(namespace_metadata())

numeric_vector <- function(value, label) {
  values <- unlist(value, use.names = FALSE)
  if (!is.numeric(values) || !length(values) || any(!is.finite(values))) {
    stop(label, " must contain finite numeric values")
  }
  as.numeric(values)
}
numeric_matrix <- function(value, label) {
  if (!is.list(value) || !length(value)) stop(label, " must be a nonempty row array")
  rows <- lapply(value, numeric_vector, label = label)
  widths <- lengths(rows)
  if (length(unique(widths)) != 1L) stop(label, " must be rectangular")
  matrix(unlist(rows, use.names = FALSE), nrow = length(rows), byrow = TRUE)
}
optional_matrix <- function(value, label) {
  if (is.null(value)) NULL else numeric_matrix(value, label)
}
safe_message <- function(message) {
  # Preserve diagnostic text without putting configured local paths in JSON.
  paths <- c(isolated_library, base_library, fixture_file,
             normalizePath(output_file, mustWork = FALSE))
  for (path in paths) message <- gsub(path, "<local-path>", message, fixed = TRUE)
  message
}
installed_package_metadata <- function() {
  packages <- sort(utils::installed.packages(lib.loc = isolated_library,
                                             noCache = TRUE)[, "Package"])
  setNames(lapply(packages, function(package) {
    description <- read.dcf(file.path(isolated_library, package, "DESCRIPTION"),
                            fields = c("Version", "Built"))
    built <- description[1L, "Built"]
    list(version = unname(description[1L, "Version"]),
         Built = if (is.na(built)) NULL else safe_message(unname(built)))
  }), packages)
}
identity_md5 <- function(root, files) {
  paths <- file.path(root, files)
  if (any(!file.exists(paths) | dir.exists(paths))) {
    stop("Required identity files are missing: ", paste(files, collapse = ", "))
  }
  checksums <- tools::md5sum(paths)
  if (anyNA(checksums)) stop("Could not read every required identity file")
  as.list(setNames(unname(checksums), files))
}
runtime_identity_md5 <- function() {
  # Prefer the executable used by Unix R. Fallback names remain explicit in
  # the report, including Windows console executables and the bin/R launcher.
  candidates <- c("bin/exec/R", "bin/x64/Rterm.exe", "bin/Rterm.exe",
                  "bin/x64/R.exe", "bin/R.exe", "bin/R")
  paths <- file.path(R.home(), candidates)
  available <- which(file.exists(paths) & !dir.exists(paths))
  if (!length(available)) stop("Could not locate an R runtime identity file")
  identity_md5(R.home(), candidates[available[[1L]]])
}

payload <- jsonlite::fromJSON(fixture_file, simplifyVector = FALSE)
expected_names <- c("single_exposure", "quantile_rounding_boundaries",
                    "prediction_boundaries", "adjusted_positive", "adjusted_mixed",
                    "adjusted_cancellation", "tied_endpoints", "explicit_breaks",
                    "prequantized")
case_names <- vapply(payload$cases, function(case) case$name, character(1))
if (length(case_names) != 9L || anyDuplicated(case_names) ||
    !setequal(case_names, expected_names)) {
  stop("FIXTURES_JSON must contain the nine current named Gaussian cases")
}
if (!identical(payload$reference$package, "qgcomp") ||
    !identical(payload$reference$version, "2.19.6")) {
  stop("FIXTURES_JSON must identify the qgcomp 2.19.6 reference")
}

export_case <- function(case) {
  input <- case$inputs
  X <- numeric_matrix(input$exposures, "exposures")
  C <- optional_matrix(input$covariates, "covariates")
  y <- numeric_vector(input$y, "y")
  Xnew <- numeric_matrix(input$new_exposures, "new_exposures")
  Cnew <- optional_matrix(input$new_covariates, "new_covariates")
  if (length(y) != nrow(X) || ncol(Xnew) != ncol(X) ||
      (!is.null(C) && nrow(C) != nrow(X)) ||
      xor(is.null(C), is.null(Cnew)) ||
      (!is.null(Cnew) && (nrow(Cnew) != nrow(Xnew) || ncol(Cnew) != ncol(C)))) {
    stop(case$name, ": input dimensions are inconsistent")
  }
  expnms <- colnames(X) <- colnames(Xnew) <- paste0("x", seq_len(ncol(X)))
  cn <- character(0)
  data <- as.data.frame(X)
  raw_newdata <- as.data.frame(Xnew)
  if (!is.null(C)) {
    cn <- colnames(C) <- colnames(Cnew) <- paste0("z", seq_len(ncol(C)))
    data <- cbind(data, as.data.frame(C))
    raw_newdata <- cbind(raw_newdata, as.data.frame(Cnew))
  }
  data$y <- y
  q <- input$q
  breaks <- if (is.null(input$breaks)) NULL else lapply(input$breaks, numeric_vector, label = "breaks")
  formula <- stats::reformulate(c(expnms, cn), response = "y")
  fit <- qgcomp::qgcomp.glm.noboot(
    formula, data = data, expnms = expnms, q = q, breaks = breaks,
    family = stats::gaussian(), bayes = FALSE, alpha = input$alpha
  )
  summary_fit <- summary(fit$fit)
  qdata <- if (is.null(q) && is.null(breaks)) data else {
    qgcomp::quantize(data, expnms = expnms, q = q, breaks = breaks)$data
  }
  qnew <- if (is.null(fit$breaks)) raw_newdata else {
    qgcomp::quantize(raw_newdata, expnms = expnms, q = q, breaks = fit$breaks)$data
  }
  predictions <- stats::predict(fit$fit, newdata = qnew, type = "response")
  wrapper_warnings <- character(0)
  wrapper_prediction <- tryCatch(withCallingHandlers({
    values <- stats::predict(fit, newdata = raw_newdata, type = "response")
    list(status = "ok", values = unname(values))
  }, warning = function(warning) {
    wrapper_warnings <<- c(wrapper_warnings, safe_message(conditionMessage(warning)))
    invokeRestart("muffleWarning")
  }), error = function(error) {
    list(status = "error", error = safe_message(conditionMessage(error)),
         error_class = unname(class(error)[1L]))
  })
  if (length(wrapper_warnings)) wrapper_prediction$warnings <- wrapper_warnings
  summary_warnings <- character(0)
  summary_output <- utils::capture.output({
    registered_summary <- tryCatch(withCallingHandlers({
      # Call the registered qgcompfit method, separately from summary(fit$fit).
      # Preserve the returned table even when it omits a statistic column.
      summary_result <- summary(fit)
      coefficients <- summary_result$coefficients
      if (!is.matrix(coefficients) || !is.numeric(coefficients)) {
        stop("registered summary did not return a numeric coefficient matrix")
      }
      list(status = "ok", row_names = as.list(unname(rownames(coefficients))),
           column_names = as.list(unname(colnames(coefficients))),
           values = unname(coefficients))
    }, warning = function(warning) {
      summary_warnings <<- c(summary_warnings, safe_message(conditionMessage(warning)))
      invokeRestart("muffleWarning")
    }), error = function(error) {
      list(status = "error", error = safe_message(conditionMessage(error)),
           error_class = unname(class(error)[1L]))
    })
  })
  registered_summary$output <- as.list(unname(vapply(summary_output, safe_message, character(1))))
  registered_summary$warnings <- as.list(unname(summary_warnings))
  list(
    name = case$name,
    column_names = list(
      coefficients = as.list(unname(names(stats::coef(fit$fit)))),
      exposures = as.list(unname(expnms))
    ),
    expected = list(
      coefficients = unname(stats::coef(fit$fit)),
      covariance = unname(summary_fit$cov.scaled),
      scores = unname(as.matrix(qdata[, expnms, drop = FALSE])),
      breaks = fit$breaks, psi = unname(fit$psi), variance = unname(fit$var.psi),
      se = unname(sqrt(fit$var.psi)), ci = unname(fit$ci),
      p_value = unname(tail(fit$pval, 1L)), statistic = unname(tail(fit$tstat, 1L)),
      df_pvalue = fit$df, df_residual = fit$fit$df.residual,
      positive_psi = fit$pos.psi, negative_psi = fit$neg.psi,
      positive_weights = as.list(fit$pos.weights), negative_weights = as.list(fit$neg.weights),
      fitted = unname(stats::fitted(fit$fit)), predictions = unname(predictions)
    ),
    wrapper_prediction = wrapper_prediction,
    registered_summary = registered_summary
  )
}

cases <- lapply(payload$cases, export_case)
# These checksums identify the inspected files; they do not authenticate them.
installed_package_md5 <- identity_md5(
  qgcomp_location, c("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")
)
r_runtime_md5 <- runtime_identity_md5()
metadata <- namespace_metadata()
reference <- c(list(
  package = "qgcomp", version = as.character(getNamespaceVersion("qgcomp")),
  mode = "installed package", R_version = R.version.string, platform = R.version$platform,
  library_verified = qgcomp_library_verified,
  installed_packages = installed_package_metadata(),
  installed_package_md5 = installed_package_md5,
  r_runtime_md5 = r_runtime_md5
), metadata)
jsonlite::write_json(list(reference = reference, cases = cases), output_file,
                     pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")
cat(length(cases), "installed-package Gaussian cases exported; wrapper outcomes recorded separately\n")
