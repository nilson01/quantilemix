# SPDX-License-Identifier: GPL-2.0-or-later
# Same-input Gaussian comparison; requires an isolated supported qgcomp library.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) stop("Expected LIBRARY DEPENDENCIES VERSION INPUT_JSON OUTPUT_JSON")
lib <- normalizePath(args[[1L]], mustWork = TRUE)
dep_lib <- normalizePath(args[[2L]], mustWork = TRUE)
expected_version <- args[[3L]]
if (!(expected_version %in% c("1.3.0", "2.19.6"))) stop("Unsupported qgcomp version")
base_lib <- normalizePath(.Library, mustWork = TRUE)
.libPaths(c(lib, dep_lib, base_lib), include.site = FALSE)
if (!setequal(.libPaths(), unique(c(lib, dep_lib, base_lib)))) stop("R library isolation failed")
if (as.character(utils::packageVersion("qgcomp", lib.loc = lib)) != expected_version) {
  stop("Wrong installed qgcomp version")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite required")
if (!requireNamespace("qgcomp", lib.loc = lib, quietly = TRUE)) stop("qgcomp required")
if (normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")) !=
    normalizePath(file.path(lib, "qgcomp"))) stop("Wrong qgcomp origin")
input <- jsonlite::fromJSON(args[[4L]], simplifyVector = FALSE)
cases <- lapply(input$cases, function(case) {
  x <- do.call(rbind, lapply(case$exposures, unlist, use.names = FALSE))
  if (ncol(x) != 4L || nrow(x) != 500L) stop("Unexpected exposure shape")
  colnames(x) <- paste0("x", 1:4)
  y <- as.numeric(unlist(case$y, use.names = FALSE))
  if (length(y) != 500L || any(!is.finite(y))) stop("Invalid outcome")
  dat <- data.frame(x, y = y)
  fit_fun <- if (expected_version == "1.3.0") qgcomp::qgcomp.noboot else qgcomp::qgcomp.glm.noboot
  result <- fit_fun(
    y ~ x1 + x2 + x3 + x4, expnms = paste0("x", 1:4),
    data = dat, q = NULL, family = gaussian(), alpha = 0.05)
  list(scenario = case$scenario, repetition = case$repetition,
       psi = unname(as.numeric(result$psi)),
       variance = unname(as.numeric(result$var.psi)))
})
package_files <- file.path(lib, "qgcomp", c("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"))
if (any(!file.exists(package_files))) stop("Installed package identity files missing")
package_md5 <- as.list(unname(tools::md5sum(package_files)))
names(package_md5) <- c("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")
imports <- c("arm", "future", "future.apply", "ggplot2", "gridExtra", "jsonlite")
dependency_versions <- as.list(vapply(imports, function(name) {
  as.character(utils::packageVersion(name))
}, character(1)))
output <- list(reference = paste("installed R qgcomp", expected_version), R_version = as.character(getRversion()),
               package_origin_verified = TRUE, installed_package_md5 = package_md5,
               dependency_versions = dependency_versions,
               cases = cases)
jsonlite::write_json(output, args[[5L]], auto_unbox = TRUE, digits = NA, pretty = TRUE)
