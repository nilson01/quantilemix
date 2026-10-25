#!/usr/bin/env Rscript
lib <- Sys.getenv("QGCOMP_R_LIB", "../.r-library-qgcomp-2.19.6")
.libPaths(c(lib, .libPaths()))
stopifnot(as.character(packageVersion("qgcomp")) == "2.19.6")
library(jsonlite)
cov <- matrix(c(.010051348, -.0039332248, -.0036965571,
                -.0039332248, .0051807876, .0007706792,
                -.0036965571, .0007706792, .0050996587), nrow=3, byrow=TRUE)
dimnames(cov) <- list(c("(Intercept)", "x1", "x2"), c("(Intercept)", "x1", "x2"))
out <- list(reference_version="2.19.6", names=colnames(cov), covariance=unname(cov),
            se_default=qgcomp::se_comb(c("x1", "x2"), cov),
            se_scalar=qgcomp::se_comb(c("x1", "x2"), cov, grad=.5),
            se_full=qgcomp::se_comb(c("x1", "x2"), cov, grad=c(0, .3, .1)),
            vc_default=unname(qgcomp::vc_comb("(Intercept)", c("x1", "x2"), cov)),
            vc_scalar_r=unname(qgcomp::vc_comb("(Intercept)", c("x1", "x2"), cov, grad=.5)))
write_json(out, "tests/fixtures/r_covariance_helpers_2_19_6.json",
           auto_unbox=TRUE, digits=16, pretty=TRUE)
