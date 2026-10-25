# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed weighted binomial and Poisson cases from isolated installed qgcomp.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('Usage: Rscript export_glm_weights_r.R LIB OUTPUT_JSON')
lib <- normalizePath(args[[1L]], mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib, normalizePath(.Library,mustWork=TRUE)), include.site=FALSE)
if (as.character(utils::packageVersion('qgcomp',lib.loc=lib)) != '2.19.6') stop('qgcomp 2.19.6 required')
if (!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) || !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if (!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
               normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
make_case <- function(family, d, q) {
  formula <- if ('z' %in% names(d)) y~x1+x2+z else y~x1+x2
  fit <- qgcomp::qgcomp.glm.noboot(formula, data=d, expnms=c('x1','x2'),
    q=q, family=if (family=='binomial') stats::binomial() else stats::poisson(),
    weights=w)
  new <- d[c(1L, 3L, 8L), , drop=FALSE]
  newq <- if (is.null(q)) new else qgcomp::quantize(new,c('x1','x2'),q=q,breaks=fit$breaks)$data
  scored <- if (is.null(q)) d else qgcomp::quantize(d,c('x1','x2'),q=q)$data
  list(name=paste(family,if(is.null(q)) 'scores' else 'quantized',sep='_'),
       family=family,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=if('z' %in% names(d)) unname(d$z) else NULL,
                   weights=unname(d$w),q=q,
                   new_x1=unname(new$x1),new_x2=unname(new$x2),
                   new_z=if('z' %in% names(new)) unname(new$z) else NULL),
       expected=list(coefficients=unname(stats::coef(fit$fit)),
                     covariance=unname(summary(fit$fit)$cov.scaled),
                     scores=unname(as.matrix(scored[,c('x1','x2')])),
                     breaks=fit$breaks, psi=unname(fit$psi),
                     variance=unname(fit$var.psi), ci=unname(fit$ci),
                     p=unname(tail(fit$pval,1L)),
                     fitted=unname(stats::fitted(fit$fit)),
                     predictions=unname(stats::predict(fit$fit,newdata=newq,type='response')),
                     residual_df=fit$fit$df.residual,
                     prior_weights=unname(stats::model.weights(stats::model.frame(fit$fit)))))
}
# Four fixed cases: exact scores and raw measurements in each family.
b <- expand.grid(x1=0:1,x2=0:1,rep=1:12,KEEP.OUT.ATTRS=FALSE)
b$y <- as.integer((b$x1 + 2*b$x2 + 3*b$rep) %% 7 < 3)
b$w <- 1L + (seq_len(nrow(b)) %% 3L)
bq <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
bq$z <- as.numeric(bq$rep %% 3)
bq$y <- as.integer((2*bq$x1+3*bq$x2+5*bq$rep) %% 11 < (4+bq$x1))
bq$w <- 1L + (seq_len(nrow(bq)) %% 3L)
p <- b
p$y <- as.integer((2*p$x1+p$x2+3*p$rep) %% 5)
pq <- bq
pq$y <- as.integer((2*pq$x1+pq$x2+3*pq$rep) %% 5)
cases <- list(make_case('binomial',b,NULL),make_case('binomial',bq,4L),
              make_case('poisson',p,NULL),make_case('poisson',pq,4L))
payload <- list(R_version=R.version.string,qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R weighted GLM cases exported\n')
