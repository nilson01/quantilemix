# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed pointwise comparisons from isolated installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_pointwise_noboot_r.R LIBRARY OUTPUT_JSON")
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
  stop("qgcomp/jsonlite or dependencies are missing")
}
if (!identical(normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
               normalizePath(file.path(lib, "qgcomp")))) {
  stop("loaded qgcomp namespace is outside the requested library")
}

design <- expand.grid(x1=0:3, x2=0:3, rep=1:8, KEEP.OUT.ATTRS=FALSE)
design$z <- as.numeric(design$rep %% 3)
design$gaussian <- 1.4 + .6*design$x1 - .3*design$x2 + .25*design$z +
                   .04*((design$rep*design$x1 + 2*design$x2) %% 5)
design$binomial <- as.integer((2*design$x1 + 3*design$x2 + 5*design$rep) %% 11 <
                              (4 + design$x1))
design$poisson <- as.integer((2*design$x1 + 3*design$x2 + 5*design$rep) %% 9)

make_case <- function(family, ref, alpha) {
  d <- data.frame(y=design[[family]], x1=design$x1, x2=design$x2, z=design$z)
  r_family <- switch(family, gaussian=stats::gaussian(),
                     binomial=stats::binomial(), poisson=stats::poisson())
  model <- qgcomp::qgcomp.glm.noboot(y ~ x1+x2+z, expnms=c("x1", "x2"),
                                      data=d, q=4L, family=r_family)
  bounds <- qgcomp::pointwisebound.noboot(model, alpha=alpha, pointwiseref=ref)
  list(name=paste(family, ref, sep="_ref"), family=family,
       inputs=list(y=unname(d$y), x1=unname(d$x1), x2=unname(d$x2),
                   z=unname(d$z), q=4L, pointwiseref=ref, alpha=alpha),
       expected=list(columns=names(bounds), table=lapply(bounds, unname),
                     link=attr(bounds, "link"),
                     fitted_coefficients=unname(stats::coef(model$fit)),
                     fitted_covariance=unname(stats::vcov(model$fit)),
                     joint_coefficients=unname(stats::coef(model))))
}

cases <- list(make_case("gaussian", 1L, .05), make_case("gaussian", 3L, .10),
              make_case("binomial", 1L, .05), make_case("binomial", 3L, .10),
              make_case("poisson", 1L, .05), make_case("poisson", 3L, .10))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion("qgcomp")),
                jsonlite_version=as.character(utils::packageVersion("jsonlite")),
                namespace_verified=TRUE,cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null="null")
cat(length(cases),"installed-R pointwise cases exported\n")
