# SPDX-License-Identifier: GPL-2.0-or-later
# Marginal Cox stage on identical Python-generated outcomes; installed R reference.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3L) stop('Usage: Rscript export_cox_bootstrap_stage_r.R LIBRARY INPUT_JSON OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
input <- normalizePath(args[[2L]],mustWork=TRUE)
out <- args[[3L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('survival',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing R dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
cases <- jsonlite::fromJSON(input,simplifyVector=FALSE)$cases
if(length(cases)!=2L) stop('expected two fixed cases')
make_case <- function(case){
  time <- unlist(case$simulated_time)
  event <- unlist(case$simulated_event)
  level <- unlist(case$simulated_level)
  degree <- as.integer(case$degree)
  if(anyNA(time) || anyNA(event) || anyNA(level)) stop('missing stage input')
  d <- data.frame(time=time,event=event,level=level)
  fit <- survival::coxph(survival::Surv(time,event)~poly(level,degree=degree,raw=TRUE),
                         data=d,ties='efron',x=TRUE,y=TRUE)
  effect <- unname(as.numeric(stats::coef(fit)))
  boots <- do.call(rbind,lapply(case$bootstrap_effects,unlist))
  if(nrow(boots)<2L || ncol(boots)!=degree) stop('bootstrap shape differs')
  covariance <- matrix(stats::cov(boots),degree,degree)
  se <- sqrt(diag(covariance))
  ci <- cbind(effect + se*stats::qnorm(.025),effect + se*stats::qnorm(.975))
  p <- 2*stats::pnorm(-abs(effect/se))
  list(name=case$name,degree=degree,
       expected=list(psi=effect,covariance=unname(covariance),se=se,
                     ci=unname(ci),p=p,events=sum(event)))
}
result <- list(R_version=R.version.string,
               qgcomp_version=as.character(utils::packageVersion('qgcomp')),
               survival_version=as.character(utils::packageVersion('survival')),
               cases=lapply(cases,make_case))
jsonlite::write_json(result,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(result$cases),'installed-R marginal Cox stage cases exported\n')
