# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed predictions from the exported msm.predict in installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop('Usage: Rscript export_msm_predict_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if (as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if (!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
    !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if (!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
               normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')

make_case <- function(name,d,form,family,q,seed,new_scores) {
  set.seed(seed)
  schedule <- t(replicate(24L,sort(sample(seq_len(nrow(d)),nrow(d),replace=TRUE))))
  m <- qgcomp::qgcomp.glm.boot(form,data=d,expnms=c('x1','x2'),q=q,
                                family=family,B=24L,seed=seed)
  default <- qgcomp::msm.predict(m)
  supplied <- qgcomp::msm.predict(m,newdata=data.frame(psi=new_scores))
  list(name=name,family=family$family,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=if ('z' %in% names(d)) unname(d$z) else NULL,
                   time=if ('time' %in% names(d)) unname(d$time) else NULL,
                   q=q,new_scores=new_scores,indices=unname(schedule-1L)),
       expected=list(default=unname(as.numeric(default)),
                     supplied=unname(as.numeric(supplied)),
                     levels=unname(sort(unique(m$index))),
                     msm_coefficients=unname(as.numeric(stats::coef(m$msmfit))),
                     means=unname(as.numeric(tapply(m$y.expectedmsm,m$index,mean))),
                     index=unname(as.numeric(m$index))))
}
g <- expand.grid(x1=0:2,x2=0:2,rep=1:6,KEEP.OUT.ATTRS=FALSE)
g$z <- as.numeric(g$rep %% 3L)
g$y <- 1+1.1*g$x1-0.3*g$x2+0.2*g$z+
       0.11*((2L*seq_len(nrow(g))) %% 7L-3L)
g0 <- expand.grid(x1=0:1,x2=0:1,rep=1:12,KEEP.OUT.ATTRS=FALSE)
g0$y <- 2+1.2*g0$x1-0.4*g0$x2+0.07*(g0$rep %% 5)
p <- expand.grid(x1=0:3,x2=0:3,rep=1:10,KEEP.OUT.ATTRS=FALSE)
p$z <- as.numeric(p$rep %% 3)
p$y <- as.integer((3*p$x1+2*p$x2+5*p$z+7*p$rep) %% 11+1)
p0 <- do.call(rbind,lapply(1:4,function(i)
  data.frame(x1=rep(c(0,1,0,1)[i],20),
             x2=rep(c(0,0,1,1)[i],20),
             y=rep(c(1,2,3,6)[i],20))))
o <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
o$z <- as.numeric(o$rep %% 3)
o$time <- as.numeric(1+o$rep %% 5)
o$y <- as.integer((2*o$x1+3*o$x2+5*o$rep) %% 9)
levels <- c(-0.25,0,0.5,1.5,3,4.25)
cases <- list(
  make_case('gaussian_prequantized',g0,y~x1+x2,stats::gaussian(),NULL,173L,levels),
  make_case('gaussian_adjusted',g,y~x1+x2+z,stats::gaussian(),3L,291L,levels),
  make_case('poisson_prequantized',p0,y~x1+x2,stats::poisson(),NULL,991L,levels),
  make_case('poisson_adjusted',p,y~x1+x2+z,stats::poisson(),4L,991L,levels),
  make_case('poisson_offset',o,y~x1+x2+z+offset(log(time)),stats::poisson(),4L,991L,levels))
payload <- list(R_version=R.version.string,qgcomp_version='2.19.6',
                namespace_verified=TRUE,cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R marginal prediction cases exported\n')
