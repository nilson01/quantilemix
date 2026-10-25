# SPDX-License-Identifier: GPL-2.0-or-later
# Conditional survival curves from isolated installed qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3L) stop('Usage: Rscript export_cox_curves_r.R LIBRARY INPUT_FIXTURE OUTPUT_JSON')
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
source_cases <- jsonlite::fromJSON(input,simplifyVector=FALSE)$cases
if(length(source_cases)!=2L) stop('expected two pinned Cox tie cases')
make_case <- function(case){
  a <- case$inputs
  d <- data.frame(time=unlist(a$time),event=unlist(a$event),
                  x1=unlist(a$x1),x2=unlist(a$x2),z=unlist(a$z))
  m <- qgcomp::qgcomp.cox.noboot(survival::Surv(time,event)~x1+x2+z,
                                 data=d,expnms=c('x1','x2'),q=a$q)
  rows <- c(1L,16L,29L,64L)
  newdata <- data.frame(x1=m$qx$x1[rows],x2=m$qx$x2[rows],z=d$z[rows])
  sf <- survival::survfit(m$fit,newdata=newdata,se.fit=FALSE)
  base <- survival::basehaz(m$fit,centered=FALSE)
  list(name=case$name, profile_rows=rows,
       expected=list(time=unname(sf$time),
                     baseline_time=unname(base$time),
                     baseline_cumhaz=unname(base$hazard),
                     survival=unname(sf$surv),
                     profile_scores=unname(as.matrix(newdata[,c('x1','x2')]))))
}
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                survival_version=as.character(utils::packageVersion('survival')),
                cases=lapply(source_cases,make_case))
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(payload$cases),'installed-R conditional survival curve cases exported\n')
