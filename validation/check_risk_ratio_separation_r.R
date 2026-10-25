# SPDX-License-Identifier: GPL-2.0-or-later
# Installed qgcomp 2.19.6 on the retained separated bootstrap draw.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3L) stop('Usage: Rscript SCRIPT LIBRARY DRAW_CSV OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
draw <- normalizePath(args[[2L]],mustWork=TRUE)
out <- args[[3L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link)))
  stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6')
  stop('installed qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing isolated R dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
d <- utils::read.csv(draw)
if(!identical(names(d),c('id__','x1','x2','x3','x4','z','y')) ||
   nrow(d)!=500L || sum(d$z==0L)!=255L ||
   sum(d$y[d$z==0L])!=0L || sum(d$y[d$z==1L])!=32L)
  stop('draw identity or separation counts differ')

# qgcomp.glm.boot calls this installed namespace function with main=FALSE for
# each resample. Supplying the retained draw checks that exact internal step.
capture <- function(expr) {
  seen <- character()
  value <- tryCatch(withCallingHandlers(expr,warning=function(w) {
    seen <<- c(seen,conditionMessage(w)); invokeRestart('muffleWarning')
  }),error=function(e)e)
  list(value=value,warnings=seen)
}
internal <- capture(get('msm_fit',envir=asNamespace('qgcomp'))(
  y~x1+x2+x3+x4+z,d,intvals=0:1,expnms=c('x1','x2','x3','x4'),
  rr=TRUE,main=FALSE,degree=1,id='id__',bayes=FALSE,MCsize=nrow(d),
  hasintercept=TRUE,family=stats::binomial()))
if(inherits(internal$value,'error')) stop(conditionMessage(internal$value))
fit <- internal$value

# A public call on the same separated input checks exported behavior. Its four
# fresh resamples are not the original Python study's 200-row schedule.
public <- capture(qgcomp::qgcomp.glm.boot(
  y~x1+x2+x3+x4+z,data=d,expnms=c('x1','x2','x3','x4'),q=NULL,
  family=stats::binomial(),rr=TRUE,B=4L,seed=451L))
if(inherits(public$value,'error')) stop(conditionMessage(public$value))
m <- public$value
payload <- list(
  R_version=R.version.string,
  qgcomp_version=as.character(utils::packageVersion('qgcomp')),
  jsonlite_version=as.character(utils::packageVersion('jsonlite')),
  draw=list(rows=nrow(d),unique_original_ids=length(unique(d$id__)),
            events=sum(d$y),z0_rows=sum(d$z==0L),z0_events=sum(d$y[d$z==0L]),
            z1_events=sum(d$y[d$z==1L])),
  internal=list(conditional_converged=fit$fit$converged,
                conditional_iterations=fit$fit$iter,
                conditional_coefficients=unname(as.numeric(stats::coef(fit$fit))),
                msm_converged=fit$msmfit$converged,
                msm_coefficients=unname(as.numeric(stats::coef(fit$msmfit))),
                warnings=internal$warnings),
  public=list(B=4L,psi=unname(as.numeric(m$psi)),
              se=unname(as.numeric(sqrt(m$var.psi))),
              ci=unname(as.numeric(m$ci)),
              conditional_converged=m$fit$converged,
              conditional_iterations=m$fit$iter,
              conditional_coefficients=unname(as.numeric(stats::coef(m$fit))),
              warnings=public$warnings))
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat('Exact separated draw checked in installed R qgcomp 2.19.6\n')
