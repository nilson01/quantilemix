# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed marginal odds-ratio cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_odds_ratio_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]], mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib, normalizePath(.Library, mustWork=TRUE)), include.site=FALSE)
stopifnot(as.character(utils::packageVersion('qgcomp', lib.loc=lib))=='2.19.6')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('jsonlite'),'path')),
              normalizePath(file.path(lib,'jsonlite')))) stop('wrong jsonlite origin')
make_case <- function(name,d,form) {
  B <- 24L; seed <- 451L
  set.seed(seed)
  schedule <- t(replicate(B, sort(sample(seq_len(nrow(d)), nrow(d), replace=TRUE))))
  m <- qgcomp::qgcomp.glm.boot(form, data=d, expnms=c('x1','x2'), q=4L,
                               family=stats::binomial(), B=B, seed=seed, rr=FALSE)
  if(!identical(as.integer(sort(unique(m$index))),0:3)) stop('wrong intervention levels')
  list(name=name,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=if('z' %in% names(d)) unname(d$z) else NULL,
                   scores=unname(as.matrix(m$qx)),
                   indices=unname(schedule-1L)),
       expected=list(psi=unname(m$psi),se=unname(sqrt(m$var.psi)),
                     ci=unname(m$ci),p=unname(tail(m$pval,1L)),
                     risks=unname(as.numeric(tapply(m$y.expected,m$index,mean))),
                     msm_coefficients=unname(as.numeric(m$msmfit$coefficients)),
                     msm_fitted_risks=unname(as.numeric(tapply(m$y.expectedmsm,m$index,mean))),
                     effects=unname(as.numeric(m$bootsamps[2L,]))))
}
d <- expand.grid(x1=0:3,x2=0:3,rep=1:20,KEEP.OUT.ATTRS=FALSE)
d$z <- (d$rep %% 3)/2
d$y <- as.integer(((7*d$rep+3*d$x1+5*d$x2) %% 23)/23 <
                  .16+.06*(d$x1+d$x2)+.05*d$z)
cases <- list(make_case('four_levels_unadjusted',d[,c('x1','x2','y')],y~x1+x2),
              make_case('four_levels_adjusted',d[,c('x1','x2','z','y')],y~x1+x2+z))
jsonlite::write_json(list(R_version=R.version.string,
                          qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                          jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                          cases=cases),out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R marginal odds-ratio cases exported\n')
