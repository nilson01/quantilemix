# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed pointwise bootstrap tables from isolated installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop('Usage: Rscript export_pointwise_boot_r.R LIBRARY OUTPUT_JSON')
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

make_case <- function(name,d,form,family,q,seed,ref) {
  B <- 24L
  set.seed(seed)
  schedule <- t(replicate(B,sort(sample(seq_len(nrow(d)),nrow(d),replace=TRUE))))
  m <- qgcomp::qgcomp.glm.boot(form,data=d,expnms=c('x1','x2'),q=q,
                                family=family,B=B,seed=seed)
  pw <- qgcomp::pointwisebound.boot(m,pointwiseref=ref,alpha=0.05)
  list(name=name,family=family$family,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=if ('z' %in% names(d)) unname(d$z) else NULL,
                   time=if ('time' %in% names(d)) unname(d$time) else NULL,
                   q=q,indices=unname(schedule-1L),pointwiseref=ref,alpha=0.05),
       expected=list(columns=names(pw),link=attr(pw,'link'),
                     table=lapply(pw,unname),
                     means=unname(as.numeric(tapply(m$y.expectedmsm,m$index,mean))),
                     covariance=unname(m$cov.yhat)))
}
g <- expand.grid(x1=0:2,x2=0:2,rep=1:6,KEEP.OUT.ATTRS=FALSE)
g$z <- as.numeric(g$rep %% 3L)
g$y <- 1+1.1*g$x1-0.3*g$x2+0.2*g$z+
       0.11*((2L*seq_len(nrow(g))) %% 7L-3L)
p <- expand.grid(x1=0:3,x2=0:3,rep=1:10,KEEP.OUT.ATTRS=FALSE)
p$z <- as.numeric(p$rep %% 3)
p$y <- as.integer((3*p$x1+2*p$x2+5*p$z+7*p$rep) %% 11+1)
o <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
o$z <- as.numeric(o$rep %% 3)
o$time <- as.numeric(1+o$rep %% 5)
o$y <- as.integer((2*o$x1+3*o$x2+5*o$rep) %% 9)
cases <- list(
  make_case('gaussian_ref1',g,y~x1+x2+z,stats::gaussian(),3L,291L,1L),
  make_case('gaussian_ref3',g,y~x1+x2+z,stats::gaussian(),3L,291L,3L),
  make_case('poisson_ref1',p,y~x1+x2+z,stats::poisson(),4L,991L,1L),
  make_case('poisson_ref4',p,y~x1+x2+z,stats::poisson(),4L,991L,4L),
  make_case('offset_ref1',o,y~x1+x2+z+offset(log(time)),stats::poisson(),4L,991L,1L),
  make_case('offset_ref4',o,y~x1+x2+z+offset(log(time)),stats::poisson(),4L,991L,4L))
payload <- list(R_version=R.version.string,qgcomp_version='2.19.6',
                namespace_verified=TRUE,cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R pointwise bootstrap cases exported\n')
