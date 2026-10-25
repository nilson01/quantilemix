# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed additive Gaussian bootstrap cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_bootstrap_r.R LIBRARY OUTPUT_JSON")
lib <- normalizePath(args[[1L]], mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop("OUTPUT_JSON already exists")
.libPaths(c(lib, normalizePath(.Library, mustWork=TRUE)), include.site=FALSE)
if (as.character(utils::packageVersion("qgcomp", lib.loc=lib)) != "2.19.6")
  stop("installed qgcomp 2.19.6 is required")
if (!requireNamespace("qgcomp", lib.loc=lib, quietly=TRUE) ||
    !requireNamespace("jsonlite", lib.loc=lib, quietly=TRUE))
  stop("qgcomp/jsonlite or dependencies are missing")
if (!identical(normalizePath(getNamespaceInfo(asNamespace("qgcomp"),"path")),
               normalizePath(file.path(lib,"qgcomp"))))
  stop("loaded qgcomp namespace is outside the requested library")

make_case <- function(name, d, formula, q, B, seed) {
  model <- qgcomp::qgcomp.glm.boot(formula, data=d, expnms=c("x1","x2"),
                                   q=q, family=stats::gaussian(), B=B, seed=seed)
  # qgcomp seeds immediately before its row-bootstrap loop. The main fit has
  # MCsize=n and no random draws, so these are the same sampled IDs.
  set.seed(seed)
  resamples <- t(replicate(B, sort(sample(seq_len(nrow(d)), nrow(d), replace=TRUE))))
  list(name=name,
       inputs=list(y=unname(d$y), x1=unname(d$x1), x2=unname(d$x2),
                   z=if ("z" %in% names(d)) unname(d$z) else NULL,
                   q=q, B=B, seed=seed),
       resample_indices=unname(resamples-1L),
       expected=list(
         psi=unname(model$psi), variance=unname(model$var.psi),
         se=unname(sqrt(model$var.psi)), ci=unname(model$ci),
         statistic=unname(tail(model$tstat,1L)),
         p=unname(tail(model$pval,1L)), df=unname(model$df),
         bootstrap_effects=unname(as.numeric(model$bootsamps[2L,])),
         intervention_levels=unname(sort(unique(model$index))),
         intervention_means=unname(as.numeric(tapply(model$y.expected, model$index, mean))),
         scores=unname(as.matrix(model$qx)),
         breaks=model$breaks
       ))
}

grid <- expand.grid(x1=0:2, x2=0:2, rep=1:6, KEEP.OUT.ATTRS=FALSE)
grid$y <- 2 + 1.5*grid$x1 - 0.5*grid$x2 +
          0.17*((seq_len(nrow(grid))-1L) %% 5L - 2L)
d1 <- grid[,c("y","x1","x2")]
d2 <- grid
# Continuous columns exercise one-time quantization before bootstrap draws.
d2$x1 <- d2$x1 + 0.013*d2$rep + 0.001*seq_len(nrow(d2))
d2$x2 <- d2$x2 + 0.017*d2$rep + 0.0007*rev(seq_len(nrow(d2)))
d2$z <- as.numeric(d2$rep %% 3L)
d2$y <- 1 + 1.1*d2$x1 - 0.3*d2$x2 + 0.2*d2$z +
        0.11*((2L*seq_len(nrow(d2))) %% 7L - 3L)
d2 <- d2[,c("y","x1","x2","z")]
cases <- list(make_case("bootstrap_prequantized",d1,y~x1+x2,NULL,24L,173L),
              make_case("bootstrap_quantized_adjusted",d2,y~x1+x2+z,3L,24L,291L))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion("qgcomp")),
                jsonlite_version=as.character(utils::packageVersion("jsonlite")),
                qgcomp_origin=normalizePath(getNamespaceInfo(asNamespace("qgcomp"),"path")),
                library=lib,cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null="null")
cat(length(cases),"installed-R Gaussian bootstrap cases exported\n")
