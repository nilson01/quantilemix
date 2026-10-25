# SPDX-License-Identifier: GPL-2.0-or-later
# Restricted Gaussian quadratic-response cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_quadratic_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) || !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
make_case <- function(name,d,q,seed,B){
  m <- qgcomp::qgcomp.glm.boot(y~x1+x2+I(x1^2)+I(x2^2)+z,
    data=d,expnms=c('x1','x2'),q=q,family=stats::gaussian(),degree=2,B=B,seed=seed)
  set.seed(seed)
  indices <- t(replicate(B,sort(sample(seq_len(nrow(d)),nrow(d),replace=TRUE))))-1L
  list(name=name,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=unname(d$z),q=q,B=B,seed=seed),
       resample_indices=unname(indices),
       expected=list(conditional_coefficients=unname(stats::coef(m$fit)),
                     conditional_covariance=unname(stats::vcov(m$fit)),
                     fitted=unname(stats::fitted(m$fit)),
                     msm_coefficients=unname(stats::coef(m$msmfit)),
                     psi=unname(m$psi),
                     covariance=unname(m$covmat.psi),
                     variance=unname(m$var.psi),
                     se=unname(sqrt(m$var.psi)),
                     ci=unname(m$ci),
                     statistic=unname(m$tstat[2:3]),
                     p=unname(m$pval[2:3]),
                     df=unname(m$df),
                     bootstrap_coefficients=unname(t(m$bootsamps[2:3,,drop=FALSE])),
                     levels=unname(sort(unique(m$index))),
                     means=unname(as.numeric(tapply(m$y.expected,m$index,mean))),
                     scores=unname(as.matrix(m$qx)),
                     breaks=m$breaks))
}
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:5,KEEP.OUT.ATTRS=FALSE)
grid$z <- as.numeric(grid$rep %% 3)
grid$y <- 1+0.4*grid$x1-0.2*grid$x2+0.3*grid$x1^2-0.1*grid$x2^2+
          0.5*grid$z+0.05*((seq_len(nrow(grid))*7)%%9-4)
d1 <- grid[,c('y','x1','x2','z')]
d2 <- grid
d2$x1 <- d2$x1+0.017*d2$rep+0.001*seq_len(nrow(d2))
d2$x2 <- d2$x2+0.013*d2$rep+0.0007*rev(seq_len(nrow(d2)))
d2$y <- 1+0.4*d2$x1-0.2*d2$x2+0.3*d2$x1^2-0.1*d2$x2^2+
        0.5*d2$z+0.05*((seq_len(nrow(d2))*7)%%9-4)
d2 <- d2[,c('y','x1','x2','z')]
cases <- list(make_case('quadratic_prequantized',d1,NULL,173L,12L),
              make_case('quadratic_quantized',d2,4L,291L,12L))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R quadratic cases exported\n')
