# SPDX-License-Identifier: GPL-2.0-or-later
# Installed R helper on the same conditional rows and Python synthetic outcomes.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=4L) stop('Usage: Rscript export_cox_survcurve_r.R LIBRARY SOURCE_JSON STAGE_JSON OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
source <- normalizePath(args[[2L]],mustWork=TRUE)
stage <- normalizePath(args[[3L]],mustWork=TRUE)
out <- args[[4L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('survival',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing R dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
original <- jsonlite::fromJSON(source,simplifyVector=FALSE)$cases[[2L]]$inputs
synthetic <- jsonlite::fromJSON(stage,simplifyVector=FALSE)$cases[[2L]]
if(!identical(synthetic$name,'cox_quantized_efron_ties')) stop('wrong stage case')
rows <- data.frame(time=unlist(original$time),event=unlist(original$event),
                   x1=unlist(original$x1),x2=unlist(original$x2),z=unlist(original$z))
cuts <- function(v) as.numeric(stats::quantile(v,probs=0:4/4,type=7))
score <- function(v) as.numeric(cut(v,breaks=c(-1e64,cuts(v)[2:4],1e64),
                                     include.lowest=TRUE))-1
rows$x1 <- score(rows$x1)
rows$x2 <- score(rows$x2)
conditional <- survival::coxph(survival::Surv(time,event)~x1+x2+z,data=rows,
                                ties='efron',x=TRUE,y=TRUE)
sim <- data.frame(time=unlist(synthetic$simulated_time),
                  event=unlist(synthetic$simulated_event),
                  psi=unlist(synthetic$simulated_level))
sim$psi1 <- sim$psi
sim$psi2 <- sim$psi^2
encode <- function(df){
  values <- list(time=unname(as.numeric(df$time)),surv=unname(as.numeric(df$surv)))
  if('q' %in% names(df)) values$q <- unname(as.numeric(df$q))
  values
}
one <- function(degree){
  if(degree==1L){
    marginal <- survival::coxph(survival::Surv(time,event)~psi,data=sim,
                                ties='efron',x=TRUE,y=TRUE)
  } else {
    marginal <- survival::coxph(survival::Surv(time,event)~psi1+psi2,data=sim,
                                ties='efron',x=TRUE,y=TRUE)
  }
  obj <- list(q=4L,fit=conditional,msmfit=marginal,expnms=c('x1','x2'))
  curves <- qgcomp::qgcomp.survcurve.boot(obj)
  list(degree=degree,psi=unname(as.numeric(stats::coef(marginal))),
       curves=lapply(curves,encode))
}
result <- list(R_version=R.version.string,
               qgcomp_version=as.character(utils::packageVersion('qgcomp')),
               survival_version=as.character(utils::packageVersion('survival')),
               cases=lapply(1:2,one))
jsonlite::write_json(result,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat('Two installed-R marginal survival curve cases exported\n')
