# SPDX-License-Identifier: GPL-2.0-or-later
# Execute unmodified pinned R reference functions on fixed software fixtures.
# This is source-level reference validation, not a full installed-package test.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop("Usage: Rscript generate_r_fixtures.R SOURCE_DIR OUTPUT_JSON")
src <- normalizePath(args[[1]])
out <- args[[2]]
stopifnot(read.dcf(file.path(src, "DESCRIPTION"))[1, "Version"] == "2.19.6")
expected_md5 <- c("853fcb3a29f9477bcc68e1b8e588ca44", "8b3aafbeb2f51bb26757634feaf5f36b")
stopifnot(identical(unname(tools::md5sum(file.path(src,
  c("R/base.R", "R/base_utility.R")))), expected_md5))
source(file.path(src, "R/base_utility.R"))
source(file.path(src, "R/base.R"))

fixture <- function(name, X, C=NULL, q=4L, breaks=NULL, beta=NULL, alpha=.05, Xnew=NULL) {
  X <- as.matrix(X)
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  dat <- as.data.frame(X)
  cn <- character(0)
  if (!is.null(C)) {
    C <- as.matrix(C)
    cn <- colnames(C) <- paste0("z", seq_len(ncol(C)))
    dat <- cbind(dat, as.data.frame(C))
  }
  expnms <- colnames(X)
  qdat <- if (is.null(q) && is.null(breaks)) dat else quantize(dat, expnms, q, breaks)$data
  Q <- as.matrix(qdat[, expnms, drop=FALSE])
  if (is.null(beta)) beta <- seq(.3, length.out=ncol(X), by=.2)
  n <- nrow(X)
  y <- as.vector(1.25 + Q %*% beta + sin(seq_len(n) * 1.7) / 2)
  if (!is.null(C)) y <- y + as.vector(C %*% seq(.2, length.out=ncol(C), by=-.35))
  dat$y <- y
  form <- reformulate(c(expnms, cn), response="y")
  fit <- qgcomp.glm.noboot(form, data=dat, expnms=expnms, q=q,
                          breaks=breaks, family=gaussian(), bayes=FALSE, alpha=alpha)
  sm <- summary(fit$fit)
  if (is.null(Xnew)) Xnew <- X[seq_len(min(7L, n)), , drop=FALSE]
  newdat <- as.data.frame(Xnew)
  colnames(newdat) <- expnms
  Cnew <- if(is.null(C)) NULL else C[seq_len(nrow(Xnew)), , drop=FALSE]
  if(!is.null(Cnew)) newdat <- cbind(newdat, as.data.frame(Cnew))
  qnew <- if (is.null(fit$breaks)) newdat else quantize(newdat, expnms, q, fit$breaks)$data
  # stats::predict on the fitted GLM with frozen cuts isolates numerical prediction.
  pred <- predict(fit$fit, newdata=qnew, type="response")
  named <- function(v) as.list(v)
  list(name=name,
       inputs=list(y=y, exposures=unname(X), covariates=if(is.null(C)) NULL else unname(C),
                   q=q, breaks=breaks, alpha=alpha, new_exposures=unname(Xnew),
                   new_covariates=if(is.null(Cnew)) NULL else unname(Cnew)),
       expected=list(coefficients=unname(coef(fit$fit)), covariance=unname(sm$cov.scaled),
                     scores=unname(Q), breaks=fit$breaks, psi=unname(fit$psi),
                     variance=unname(fit$var.psi), se=unname(sqrt(fit$var.psi)),
                     ci=unname(fit$ci), p_value=unname(tail(fit$pval,1)),
                     statistic=unname(tail(fit$tstat,1)), df_pvalue=fit$df,
                     df_residual=fit$fit$df.residual, positive_psi=fit$pos.psi,
                     negative_psi=fit$neg.psi, positive_weights=named(fit$pos.weights),
                     negative_weights=named(fit$neg.weights), fitted=unname(fitted(fit$fit)),
                     predictions=unname(pred)))
}

i <- 0:63
X <- cbind(i %% 8 + i/1000, (3*i) %% 11 + .3*(i %% 8), (7*i) %% 13 + i/500)
C <- cbind(cos(i*.4), (i %% 5)/3)
cases <- list(
  fixture("single_exposure", matrix(seq(-3, 4, length.out=24), ncol=1)),
  fixture("quantile_rounding_boundaries", matrix(0:49, ncol=1), q=49L),
  fixture("prediction_boundaries", matrix(0:7,ncol=1),
          Xnew=matrix(c(-5,1.75,1.75+1e-10,3.5,5.25,7,20),ncol=1)),
  fixture("adjusted_positive", X, C, beta=c(.2,.4,.3)),
  fixture("adjusted_mixed", X, C, beta=c(.6,-.5,.2), alpha=.1),
  fixture("adjusted_cancellation", X, C, beta=c(.8,-.5,-.3)),
  fixture("tied_endpoints", cbind(rep(c(0,0,0,1),each=8), rep(0:7,4)), beta=c(.4,-.2)),
  fixture("explicit_breaks", X[,1:2], C[,1,drop=FALSE], q=NULL,
          breaks=list(c(-10,1.5,4.5,20),c(-10,3,8,20))),
  fixture("prequantized", as.matrix(expand.grid(x1=0:3,x2=0:3)), q=NULL, beta=c(2,-2))
)
payload <- list(reference=list(package="qgcomp", version="2.19.6",
  mode="unmodified source functions; not installed package",
  R_version=R.version.string, jsonlite_version=as.character(packageVersion("jsonlite")),
  source_md5=as.list(setNames(tools::md5sum(file.path(src,c("R/base.R","R/base_utility.R"))), c("R/base.R","R/base_utility.R")))),
  cases=cases)
dir.create(dirname(out),recursive=TRUE,showWarnings=FALSE)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null="null")
cat(length(cases), "fixed reference fixtures saved to", out, "\n")
