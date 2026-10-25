# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed weighted cluster estimating-equation cases from installed qgcomp.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_weighted_ee_r.R LIBRARY OUTPUT_JSON')
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
make_case <- function(name,d,family_obj,rr=TRUE){
  if(family_obj$family=='gaussian'){
    m <- qgcomp::qgcomp.glm.ee(y~x1+x2+z,data=d,expnms=c('x1','x2'),
        q=4L,id='id',weights=w,family=stats::gaussian(),rr=rr,degree=1,
        seed=137L,verbose=FALSE)
  } else if(family_obj$family=='poisson'){
    m <- qgcomp::qgcomp.glm.ee(y~x1+x2+z,data=d,expnms=c('x1','x2'),
        q=4L,id='id',weights=w,family=stats::poisson(),rr=rr,degree=1,
        seed=137L,verbose=FALSE)
  } else {
    m <- qgcomp::qgcomp.glm.ee(y~x1+x2+z,data=d,expnms=c('x1','x2'),
        q=4L,id='id',weights=w,family=stats::binomial(),rr=rr,degree=1,
        seed=137L,verbose=FALSE)
  }
  levels <- unname(sort(unique(m$index)))
  standardized <- vapply(levels,function(a){
    selected <- m$index==a
    stats::weighted.mean(m$y.expected[selected],d$w)
  },numeric(1L))
  list(name=name,family=family_obj$family,rr=rr,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=unname(d$z),id=unname(d$id),w=unname(d$w),q=4L),
       expected=list(conditional_coefficients=unname(m$fit$est),
                     conditional_covariance=unname(m$fit$vcov),
                     msm_coefficients=unname(m$msmfit$est),
                     msm_covariance=unname(m$msmfit$vcov),
                     full_covariance=unname(m$covmat.all_robust),
                     bread=unname(m$bread),meat=unname(m$meat),
                     psi=unname(m$psi),variance=unname(m$var.psi),
                     ci=unname(m$ci),
                     statistic=unname(if(family_obj$family=='gaussian') m$tstat else m$zstat),
                     p=unname(m$pval),df=unname(m$df),
                     scores=unname(as.matrix(m$qx)),breaks=m$breaks,
                     levels=levels,standardized=unname(standardized)))
}
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:6,KEEP.OUT.ATTRS=FALSE)
grid$id <- ceiling(seq_len(nrow(grid))/3)
grid$z <- as.numeric((grid$rep+grid$x1)%%3)
grid$w <- as.numeric(1+(seq_len(nrow(grid))*7+grid$id)%%4)
grid$x1 <- grid$x1+0.017*grid$rep+0.001*seq_len(nrow(grid))
grid$x2 <- grid$x2+0.013*grid$rep+0.0007*rev(seq_len(nrow(grid)))
gaussian <- grid
gaussian$y <- 1+0.4*grid$x1-0.2*grid$x2+0.5*grid$z+
              0.12*((grid$id*7)%%5-2)+0.03*((seq_len(nrow(grid))*11)%%13-6)
poisson <- grid
poisson$y <- as.integer((1+grid$x1+2*grid$x2+grid$z+(grid$id%%3))%%7)
binary <- grid
u <- ((seq_len(nrow(grid))*17+grid$id*7+grid$rep*3)%%97)/97
binary$y <- as.integer(u<stats::plogis(-1.3+0.22*grid$x1+0.15*grid$x2+0.4*grid$z))
cases <- list(make_case('weighted_gaussian_ee',gaussian,stats::gaussian()),
              make_case('weighted_poisson_ee',poisson,stats::poisson()),
              make_case('weighted_binomial_ee_or',binary,stats::binomial(),FALSE),
              make_case('weighted_binomial_ee_rr',binary,stats::binomial(),TRUE))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R weighted EE cases exported\n')
