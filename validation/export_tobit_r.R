# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed Gaussian censored-outcome cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_tobit_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('AER',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('survival',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
make_case <- function(name,d,q,left,right){
  m <- qgcomp::qgcomp.tobit.noboot(y~x1+x2+z,data=d,
          expnms=c('x1','x2'),q=q,left=left,right=right,alpha=0.05,
          dist='gaussian')
  fit <- m$fit
  idx <- if(length(m$psi)>1L) 2L else 1L
  list(name=name,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=unname(d$z),q=q,left=left,
                   right=if(is.finite(right)) right else NULL),
       expected=list(coefficients=unname(stats::coef(fit)),
                     covariance=unname(fit$var),sigma=unname(fit$scale),
                     psi=unname(m$psi[idx]),variance=unname(m$var.psi[idx]),
                     ci=unname(m$ci[idx,]),p=unname(m$pval[idx]),
                     scores=unname(as.matrix(m$qx)),breaks=m$breaks,
                     left_count=sum(d$y<=left),
                     right_count=if(is.finite(right)) sum(d$y>=right) else 0L))
}
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:10,KEEP.OUT.ATTRS=FALSE)
i <- seq_len(nrow(grid))
grid$z <- as.numeric(grid$rep %% 4)
u <- stats::qnorm(((i*73)%%163 + 0.5)/163)
latent <- -0.6+0.25*grid$x1+0.32*grid$x2+0.15*grid$z+0.9*u
d1 <- grid
d1$y <- pmax(latent,0)
d1 <- d1[,c('y','x1','x2','z')]
d2 <- grid
d2$x1 <- d2$x1+0.012*d2$rep+0.0001*i
d2$x2 <- d2$x2+0.008*d2$rep+0.00005*i
d2$y <- pmin(pmax(latent,0),2)
d2 <- d2[,c('y','x1','x2','z')]
cases <- list(make_case('tobit_left_prequantized',d1,NULL,0,Inf),
              make_case('tobit_two_sided_quantized',d2,4L,0,2))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                AER_version=as.character(utils::packageVersion('AER')),
                survival_version=as.character(utils::packageVersion('survival')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R Tobit cases exported\n')
