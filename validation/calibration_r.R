# SPDX-License-Identifier: GPL-2.0-or-later
# Installed-package reference adapter for at most twenty prequantized Gaussian
# cases. Empty input establishes identity only. This script installs nothing.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: Rscript --vanilla validation/calibration_r.R LIBRARY INPUT_JSON OUTPUT_JSON")
}
isolated_library <- normalizePath(args[[1L]], mustWork = TRUE)
input_file <- normalizePath(args[[2L]], mustWork = TRUE)
output_file <- args[[3L]]
if (!dir.exists(isolated_library)) stop("LIBRARY must be an existing directory")
refuse_existing_output <- function() {
  link <- Sys.readlink(output_file)
  if (file.exists(output_file) || (!is.na(link) && nzchar(link))) {
    stop("OUTPUT_JSON already exists; choose a new output path")
  }
  if (identical(normalizePath(output_file, mustWork = FALSE), input_file)) {
    stop("OUTPUT_JSON must not overwrite INPUT_JSON")
  }
}
refuse_existing_output()

# Exclude site and user fallback, then audit the origin of every namespace.
base_library <- normalizePath(.Library, mustWork = TRUE)
if (identical(isolated_library, base_library)) {
  stop("The supplied isolated library must be distinct from the R base library")
}
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
library_verified <- identical(
  normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
  normalizePath(qgcomp_location)
)
if (!library_verified || as.character(getNamespaceVersion("qgcomp")) != "2.19.6") {
  stop("The loaded qgcomp namespace does not match the requested installed package")
}
jsonlite_location <- normalizePath(getNamespaceInfo(asNamespace("jsonlite"), "path"))
if (!identical(dirname(jsonlite_location), isolated_library)) {
  stop("jsonlite must load from the supplied isolated library")
}

safe_message <- function(message) {
  paths <- c(isolated_library, base_library, input_file, R.home(),
             normalizePath(output_file, mustWork = FALSE), getwd(), tempdir())
  # Longest first avoids leaving a private suffix after replacing a parent path.
  paths <- paths[order(nchar(paths), decreasing = TRUE)]
  for (path in paths) message <- gsub(path, "<local-path>", message, fixed = TRUE)
  # Additional absolute paths can arise in dependency diagnostics.
  message <- gsub("[A-Za-z]:[/\\\\][^[:space:]'\"]+", "<local-path>", message)
  gsub("(^|[[:space:]'\"])/[^[:space:]'\"]+", "\\1<local-path>", message)
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
    unexpected_loaded_namespaces = as.list(unname(unexpected))
  )
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
  candidates <- c("bin/exec/R", "bin/x64/Rterm.exe", "bin/Rterm.exe",
                  "bin/x64/R.exe", "bin/R.exe", "bin/R")
  paths <- file.path(R.home(), candidates)
  available <- which(file.exists(paths) & !dir.exists(paths))
  if (!length(available)) stop("Could not locate an R runtime identity file")
  identity_md5(R.home(), candidates[available[[1L]]])
}

payload <- jsonlite::fromJSON(input_file, simplifyVector = FALSE)
if (!is.list(payload) || !identical(names(payload), "cases") ||
    !is.list(payload$cases) || length(payload$cases) > 20L) {
  stop("INPUT_JSON must contain a cases array with at most twenty entries")
}
case_names <- vapply(payload$cases, function(case) {
  if (!is.list(case) || anyDuplicated(names(case)) ||
      !setequal(names(case), c("name", "inputs")) ||
      !is.character(case$name) || length(case$name) != 1L || !nzchar(case$name)) {
    stop("Each case must have a nonempty name and inputs")
  }
  case$name
}, character(1))
if (anyDuplicated(case_names)) stop("Case names must be unique")

# Compute all identity fields before fitting, loading the helpers they require.
# Fingerprints identify the files inspected; they are not authenticity proofs.
installed_package_md5 <- identity_md5(
  qgcomp_location, c("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")
)
r_runtime_md5 <- runtime_identity_md5()
installed_packages <- installed_package_metadata()
initial_namespaces <- namespace_metadata()
reference <- c(list(
  package = "qgcomp", version = as.character(getNamespaceVersion("qgcomp")),
  mode = "installed package", R_version = R.version.string, platform = R.version$platform,
  library_verified = library_verified, installed_packages = installed_packages,
  installed_package_md5 = installed_package_md5, r_runtime_md5 = r_runtime_md5
), initial_namespaces)

numeric_vector <- function(value, label) {
  if (!is.list(value) || !length(value) ||
      !all(vapply(value, function(x) is.numeric(x) && length(x) == 1L && is.finite(x),
                  logical(1)))) {
    stop(label, " must be a nonempty array of finite numeric values")
  }
  as.numeric(unlist(value, use.names = FALSE))
}
numeric_matrix <- function(value, label) {
  if (!is.list(value) || !length(value)) stop(label, " must be a nonempty row array")
  rows <- lapply(value, numeric_vector, label = label)
  if (length(unique(lengths(rows))) != 1L) stop(label, " must be rectangular")
  matrix(unlist(rows, use.names = FALSE), nrow = length(rows), byrow = TRUE)
}
fit_case <- function(case) {
  input <- case$inputs
  if (!is.list(input) || anyDuplicated(names(input)) ||
      !setequal(names(input), c("exposures", "covariates", "y", "q", "alpha"))) {
    stop("Inputs must contain exposures, covariates, y, q, alpha")
  }
  if (!is.null(input$q)) stop("Only prequantized cases with q=NULL are supported")
  if (!is.numeric(input$alpha) || length(input$alpha) != 1L ||
      !is.finite(input$alpha) || input$alpha <= 0 || input$alpha >= 1) {
    stop("alpha must be finite and strictly between zero and one")
  }
  X <- numeric_matrix(input$exposures, "exposures")
  C <- if (is.null(input$covariates)) NULL else numeric_matrix(input$covariates, "covariates")
  y <- numeric_vector(input$y, "y")
  if (any(X < 0 | X >= 2^53 | X != floor(X))) {
    stop("Exposure scores must be nonnegative integers below 2^53")
  }
  if (length(y) != nrow(X) || (!is.null(C) && nrow(C) != nrow(X))) {
    stop("Input row counts are inconsistent")
  }
  c <- if (is.null(C)) 0L else ncol(C)
  if (nrow(X) <= 1L + ncol(X) + c) stop("Positive residual degrees of freedom are required")
  expnms <- colnames(X) <- paste0("x", seq_len(ncol(X)))
  covnms <- character(0)
  data <- as.data.frame(X)
  if (!is.null(C)) {
    covnms <- colnames(C) <- paste0("z", seq_len(ncol(C)))
    data <- cbind(data, as.data.frame(C))
  }
  data$y <- y
  formula <- stats::reformulate(c(expnms, covnms), response = "y")
  fit <- qgcomp::qgcomp.glm.noboot(
    formula, data = data, expnms = expnms, q = NULL, breaks = NULL,
    family = stats::gaussian(link = "identity"), bayes = FALSE, alpha = input$alpha
  )
  if (fit$fit$rank != 1L + ncol(X) + c) stop("The reference design is not full rank")
  result <- list(
    psi = unname(fit$psi), variance = unname(fit$var.psi),
    se = unname(sqrt(fit$var.psi)), ci = as.list(unname(fit$ci)),
    p_value = unname(tail(fit$pval, 1L)), df_pvalue = unname(fit$df),
    df_residual = unname(fit$fit$df.residual)
  )
  scalar_fields <- setdiff(names(result), "ci")
  if (!all(vapply(result[scalar_fields], function(x) {
        is.numeric(x) && length(x) == 1L && is.finite(x)
      }, logical(1))) || length(result$ci) != 2L ||
      any(!is.finite(unlist(result$ci, use.names = FALSE)))) {
    stop("The reference fit returned nonfinite or malformed inference")
  }
  result
}
export_case <- function(case) {
  warnings <- character(0)
  record <- tryCatch(withCallingHandlers({
    list(name = case$name, status = "ok", result = fit_case(case))
  }, warning = function(warning) {
    warnings <<- c(warnings, safe_message(conditionMessage(warning)))
    invokeRestart("muffleWarning")
  }), error = function(error) {
    list(name = case$name, status = "error", error = safe_message(conditionMessage(error)),
         error_class = unname(class(error)[1L]))
  })
  record$warnings <- as.list(unname(warnings))
  record
}
cases <- lapply(payload$cases, export_case)
# The dependency snapshot must be identical for preflight and fitting batches.
# A newly loaded namespace is an explicit startup-contract failure, never omitted.
if (!identical(namespace_metadata(), initial_namespaces)) {
  stop("Loaded namespace identity changed during fitting; stable preflight identity is unavailable")
}
if (!identical(identity_md5(qgcomp_location, names(installed_package_md5)), installed_package_md5) ||
    !identical(runtime_identity_md5(), r_runtime_md5)) {
  stop("Installed package or runtime files changed during the reference batch")
}
refuse_existing_output()
jsonlite::write_json(list(reference = reference, cases = cases), output_file,
                     pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")
