# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed untied right-censored Cox cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_cox_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('survival',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
make_case <- function(name,d,q){
  m <- qgcomp::qgcomp.cox.noboot(survival::Surv(time,event)~x1+x2+z,
       data=d,expnms=c('x1','x2'),q=q,alpha=0.05)
  list(name=name,
       inputs=list(time=unname(d$time),event=unname(d$event),
                   x1=unname(d$x1),x2=unname(d$x2),z=unname(d$z),q=q),
       expected=list(coefficients=unname(stats::coef(m$fit)),
                     covariance=unname(stats::vcov(m$fit)),
                     psi=unname(m$psi),variance=unname(m$var.psi),
                     ci=unname(m$ci),p=unname(m$pval),
                     scores=unname(as.matrix(m$qx)),breaks=m$breaks,
                     events=unname(m$fit$nevent)))
}
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
i <- seq_len(nrow(grid))
grid$z <- as.numeric(grid$rep %% 3)
grid$time <- 1+((i*37)%%131)/25+0.00001*i+0.01*grid$x1-0.008*grid$x2
grid$event <- as.integer((i*17)%%7 < 5)
if(anyDuplicated(grid$time)) stop('event times are tied')
d1 <- grid[,c('time','event','x1','x2','z')]
d2 <- grid
d2$x1 <- d2$x1+0.011*d2$rep+0.0001*i
d2$x2 <- d2$x2+0.009*d2$rep+0.00005*i
d2$time <- 1+((i*47)%%137)/28+0.00002*i+0.014*d2$x1-0.006*d2$x2
if(anyDuplicated(d2$time)) stop('second-case times are tied')
d2 <- d2[,c('time','event','x1','x2','z')]
cases <- list(make_case('cox_prequantized_untied',d1,NULL),
              make_case('cox_quantized_untied',d2,4L))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                survival_version=as.character(utils::packageVersion('survival')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R Cox cases exported\n')
