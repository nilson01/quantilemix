# SPDX-License-Identifier: GPL-2.0-or-later
# Deterministic weighted Gaussian bootstrap cases from installed qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_weighted_bootstrap_r.R LIBRARY OUTPUT_JSON")
lib <- normalizePath(args[[1L]], mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link)))
  stop("OUTPUT_JSON already exists")
.libPaths(c(lib, normalizePath(.Library, mustWork=TRUE)), include.site=FALSE)
if (as.character(utils::packageVersion("qgcomp", lib.loc=lib)) != "2.19.6")
  stop("installed qgcomp 2.19.6 is required")
if (!requireNamespace("qgcomp", lib.loc=lib, quietly=TRUE) ||
    !requireNamespace("jsonlite", lib.loc=lib, quietly=TRUE))
  stop("qgcomp/jsonlite or dependencies are missing")
if (!identical(normalizePath(getNamespaceInfo(asNamespace("qgcomp"),"path")),
               normalizePath(file.path(lib,"qgcomp"))))
  stop("loaded qgcomp namespace is outside the requested library")

make_case <- function(name, d, q, B, seed) {
  model <- qgcomp::qgcomp.glm.boot(
    y ~ x1 + x2 + z, data=d, expnms=c("x1","x2"), q=q,
    family=stats::gaussian(), weights=w, B=B, seed=seed)
  set.seed(seed)
  rows <- t(replicate(B, sort(sample(seq_len(nrow(d)), nrow(d), replace=TRUE))))
  manual <- apply(rows, 1L, function(i) {
    sampled <- d[i,,drop=FALSE]
    sampled$x1 <- model$qx[i,1L]
    sampled$x2 <- model$qx[i,2L]
    direct <- stats::lm(y ~ x1 + x2 + z, data=sampled, weights=w)
    sum(stats::coef(direct)[c("x1","x2")])
  })
  effects <- as.numeric(model$bootsamps[2L,])
  if (max(abs(manual-effects)) > 1e-9)
    stop("R bootstrap effects differ from sampled-weight regression")
  means <- as.numeric(tapply(
    model$y.expected, model$index,
    function(v) stats::weighted.mean(v, d$w)
  ))
  list(name=name,
       inputs=list(y=unname(d$y), x1=unname(d$x1), x2=unname(d$x2),
                   z=unname(d$z), weights=unname(d$w), q=q, B=B, seed=seed),
       resample_indices=unname(rows-1L),
       expected=list(psi=unname(model$psi),
                     se=unname(sqrt(model$var.psi)),
                     ci=unname(as.numeric(model$ci)),
                     bootstrap_effects=unname(effects),
                     intervention_means=unname(means),
                     scores=unname(as.matrix(model$qx)),
                     breaks=model$breaks))
}

n <- 54L
d <- data.frame(x1=rep(0:2,18), x2=rep(rep(0:2,each=3),6),
                z=rep(0:2,each=18))
d$y <- 2 + 1.5*d$x1 - 0.5*d$x2 + 0.2*d$z +
       0.17*((seq_len(n)-1L) %% 5L - 2L)
d$w <- 0.6 + ((seq_len(n)-1L) %% 7L)/5
continuous <- d
continuous$x1 <- d$x1 + 0.013*seq_len(n)
continuous$x2 <- d$x2 + 0.007*rev(seq_len(n))
cases <- list(make_case("weighted_prequantized",d,NULL,12L,173L),
              make_case("weighted_quantized",continuous,3L,12L,291L))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion("qgcomp")),
                qgcomp_origin=normalizePath(getNamespaceInfo(asNamespace("qgcomp"),"path")),
                library=lib,cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null="null")
cat(length(cases),"weighted Gaussian bootstrap cases exported\n")
