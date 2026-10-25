# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed clustered binary estimating-equation cases from installed qgcomp.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_binomial_ee_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
make_case <- function(name,d,q,rr){
  m <- qgcomp::qgcomp.glm.ee(y~x1+x2+z,data=d,expnms=c('x1','x2'),
      q=q,id='id',family=stats::binomial(),rr=rr,degree=1,seed=137L,verbose=FALSE)
  levels <- unname(sort(unique(m$index)))
  msmrisks <- if(rr) exp(m$msmfit$est[[1L]] + levels*m$msmfit$est[[2L]])
              else stats::plogis(m$msmfit$est[[1L]] + levels*m$msmfit$est[[2L]])
  list(name=name,rr=rr,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=unname(d$z),id=unname(d$id),q=q),
       expected=list(conditional_coefficients=unname(m$fit$est),
                     conditional_covariance=unname(m$fit$vcov),
                     msm_coefficients=unname(m$msmfit$est),
                     msm_covariance=unname(m$msmfit$vcov),
                     full_covariance=unname(m$covmat.all_robust),
                     bread=unname(m$bread),meat=unname(m$meat),
                     psi=unname(m$psi),variance=unname(m$var.psi),
                     ci=unname(m$ci),statistic=unname(m$zstat),
                     p=unname(m$pval),scores=unname(as.matrix(m$qx)),
                     breaks=m$breaks,levels=levels,
                     risks=unname(as.numeric(tapply(m$y.expected,m$index,mean))),
                     msm_fitted_risks=unname(msmrisks)))
}
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:6,KEEP.OUT.ATTRS=FALSE)
grid$id <- ceiling(seq_len(nrow(grid))/3)
grid$z <- as.numeric((grid$rep+grid$x1)%%3)
u <- ((seq_len(nrow(grid))*17+grid$id*7+grid$rep*3)%%97)/97
d1 <- grid
d1$x1 <- as.numeric(grid$x1>=2)+0.001*seq_len(nrow(grid))
d1$x2 <- as.numeric(grid$x2>=2)+0.0007*rev(seq_len(nrow(grid)))
d1$y <- as.integer(u<stats::plogis(-1.1+0.55*(grid$x1>=2)+
                                    0.35*(grid$x2>=2)+0.45*grid$z))
d2 <- grid
d2$x1 <- grid$x1+0.017*grid$rep+0.001*seq_len(nrow(grid))
d2$x2 <- grid$x2+0.013*grid$rep+0.0007*rev(seq_len(nrow(grid)))
d2$y <- as.integer(u<stats::plogis(-1.3+0.22*grid$x1+
                                    0.15*grid$x2+0.4*grid$z))
d1 <- d1[,c('y','x1','x2','z','id')]
d2 <- d2[,c('y','x1','x2','z','id')]
cases <- list(make_case('binomial_ee_q2_or',d1,2L,FALSE),
              make_case('binomial_ee_q2_rr',d1,2L,TRUE),
              make_case('binomial_ee_q4_or',d2,4L,FALSE),
              make_case('binomial_ee_q4_rr',d2,4L,TRUE))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R binomial EE cases exported\n')
