# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed conditional Cox log-hazard offsets from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_cox_offset_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6')
  stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('survival',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')

grid <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
i <- seq_len(nrow(grid))
grid$z <- as.numeric(grid$rep %% 3)
grid$time <- 1+((i*37)%%131)/25+0.00001*i+0.01*grid$x1-0.008*grid$x2
grid$event <- as.integer((i*17)%%7 < 5)
grid$o <- log(0.5+((i*19)%%13)/10)
grid$w <- 0.5+((i*13)%%11)/10

make_case <- function(name,d,weighted){
  f <- survival::Surv(time,event)~x1+x2+z+offset(o)
  fit <- if(weighted)
    qgcomp::qgcomp.cox.noboot(f,data=d,expnms=c('x1','x2'),
                              q=NULL,alpha=0.05,weights=w)
  else
    qgcomp::qgcomp.cox.noboot(f,data=d,expnms=c('x1','x2'),q=NULL,alpha=0.05)
  base <- survival::basehaz(fit$fit,centered=FALSE)
  new <- data.frame(x1=c(0,1),x2=c(0,1),z=c(0,1),
                    o=c(log(0.5),log(2)))
  curve <- survival::survfit(fit$fit,newdata=new)
  list(name=name,
       inputs=list(time=unname(d$time),event=unname(d$event),
                   x1=unname(d$x1),x2=unname(d$x2),z=unname(d$z),
                   offset=unname(d$o),
                   weights=if(weighted) unname(d$w) else NULL,
                   prediction=list(x1=unname(new$x1),x2=unname(new$x2),
                                   z=unname(new$z),offset=unname(new$o))),
       expected=list(coefficients=unname(stats::coef(fit$fit)),
                     covariance=unname(stats::vcov(fit$fit)),
                     psi=unname(fit$psi),variance=unname(fit$var.psi),
                     ci=unname(fit$ci),p=unname(fit$pval),
                     baseline_time=unname(base$time),
                     baseline_cumulative_hazard=unname(base$hazard),
                     prediction_time=unname(curve$time),
                     prediction_survival=unname(curve$surv),
                     events=unname(fit$fit$nevent)))
}
untied <- grid[,c('time','event','x1','x2','z','o','w')]
if(anyDuplicated(untied$time)) stop('first case has tied times')
tied <- untied
tied$time <- 1+((i*7)%%19)
if(!anyDuplicated(tied$time)) stop('second case lacks tied times')
cases <- list(make_case('offset_untied',untied,FALSE),
              make_case('offset_tied',tied,FALSE),
              make_case('offset_weighted_tied',tied,TRUE))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                survival_version=as.character(utils::packageVersion('survival')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R Cox offset cases exported\n')
