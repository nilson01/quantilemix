# SPDX-License-Identifier: GPL-2.0-or-later
# Bounded installed-R qgcomp 2.19.6 marginal survival generator audit.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3L) stop('Usage: Rscript audit_cox_bootstrap_r.R LIBRARY INPUT_FIXTURE OUTPUT_JSON')
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
a <- jsonlite::fromJSON(input,simplifyVector=FALSE)$cases[[2L]]$inputs
d <- data.frame(time=unlist(a$time),event=unlist(a$event),
                x1=unlist(a$x1),x2=unlist(a$x2),z=unlist(a$z))
set.seed(42L)
m <- qgcomp::qgcomp.cox.boot(survival::Surv(time,event)~x1+x2+z,
                              data=d,expnms=c('x1','x2'),q=4L,B=2L,MCsize=128L,
                              seed=314159L)
times <- m$y.expected[,1L]
levels <- m$index
result <- list(R_version=R.version.string,
               qgcomp_version=as.character(utils::packageVersion('qgcomp')),
               survival_version=as.character(utils::packageVersion('survival')),
               setup=list(initial_seed=42L,bootstrap_seed=314159L,
                          rows=nrow(d),B=2L,MCsize=128L,degree=1L,q=4L),
               simulated_outcomes=length(times),
               missing_time_count=sum(is.na(times)),
               missing_by_level=unname(vapply(sort(unique(levels)),
                   function(level) sum(is.na(times[levels==level])), integer(1L))),
               marginal_psi=unname(m$psi),
               bootstrap_effects=unname(m$bootsamps),
               bootstrap_variance=unname(m$var.psi))
jsonlite::write_json(result,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat('Installed-R marginal survival generator audit exported\n')
