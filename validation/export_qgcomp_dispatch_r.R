# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed routes through exported qgcomp in isolated R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop('Usage: Rscript export_qgcomp_dispatch_r.R LIBRARY OUTPUT_JSON')
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

make_case <- function(name,d,form,family,q,rr=TRUE,offset=FALSE,seed=451L) {
  B <- 24L
  schedule <- NULL
  if (family$family=='binomial' && rr) {
    set.seed(seed)
    schedule <- t(replicate(B,sort(sample(seq_len(nrow(d)),nrow(d),replace=TRUE))))
  }
  if (is.null(schedule)) {
    m <- qgcomp::qgcomp(form,data=d,expnms=c('x1','x2'),q=q,family=family,rr=rr)
  } else {
    m <- qgcomp::qgcomp(form,data=d,expnms=c('x1','x2'),q=q,family=family,rr=rr,
                         B=B,seed=seed)
  }
  list(name=name,family=family$family,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=if ('z' %in% names(d)) unname(d$z) else NULL,
                   time=if ('time' %in% names(d)) unname(d$time) else NULL,
                   q=q,rr=rr,offset=offset,B=if (is.null(schedule)) NULL else B,
                   seed=if (is.null(schedule)) NULL else seed,
                   indices=if (is.null(schedule)) NULL else unname(schedule-1L)),
       expected=list(bootstrap=isTRUE(m$bootstrap),
                     psi=unname(as.numeric(m$psi)),
                     se=unname(as.numeric(sqrt(m$var.psi))),
                     ci=unname(as.numeric(m$ci)),
                     p=unname(as.numeric(tail(m$pval,1L)))))
}
g <- expand.grid(x1=0:2,x2=0:2,rep=1:6,KEEP.OUT.ATTRS=FALSE)
g$z <- as.numeric(g$rep %% 3L)
g$y <- 1+1.1*g$x1-0.3*g$x2+0.2*g$z+
       0.11*((2L*seq_len(nrow(g))) %% 7L-3L)
gc <- g
gc$z <- factor(ifelse(gc$z==0,'low',ifelse(gc$z==1,'mid','high')))
gc$y <- g$y+0.1*(gc$z=='mid')-0.2*(gc$z=='high')
bc <- data.frame(x1=c(0,1,0,1),x2=c(0,0,1,1),successes=c(8,13,12,17))
b <- do.call(rbind,lapply(seq_len(nrow(bc)),function(i)
 data.frame(x1=rep(bc$x1[i],24),x2=rep(bc$x2[i],24),
            y=c(rep(1,bc$successes[i]),rep(0,24-bc$successes[i])))))
p <- expand.grid(x1=0:3,x2=0:3,rep=1:10,KEEP.OUT.ATTRS=FALSE)
p$z <- as.numeric(p$rep %% 3)
p$y <- as.integer((3*p$x1+2*p$x2+5*p$z+7*p$rep) %% 11+1)
o <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
o$z <- as.numeric(o$rep %% 3)
o$time <- as.numeric(1+o$rep %% 5)
o$y <- as.integer((2*o$x1+3*o$x2+5*o$rep) %% 9)
cases <- list(
  make_case('gaussian_adjusted',g,y~x1+x2+z,stats::gaussian(),3L),
  make_case('gaussian_category',gc,y~x1+x2+z,stats::gaussian(),3L),
  make_case('binary_odds',b,y~x1+x2,stats::binomial(),2L,FALSE),
  make_case('binary_default_rr',b,y~x1+x2,stats::binomial(),2L,TRUE),
  make_case('poisson_adjusted',p,y~x1+x2+z,stats::poisson(),4L),
  make_case('poisson_offset_conditional',o,y~x1+x2+z+offset(log(time)),stats::poisson(),4L,
            TRUE,TRUE,991L))
payload <- list(R_version=R.version.string,qgcomp_version='2.19.6',
                namespace_verified=TRUE,cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R dispatcher cases exported\n')
