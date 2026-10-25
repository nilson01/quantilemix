# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed two-level marginal binary cases from isolated installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_risk_ratio_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
make_case <- function(name,d,form){
  B <- 24L
  seed <- 451L
  set.seed(seed)
  schedule <- t(replicate(B,sort(sample(seq_len(nrow(d)),nrow(d),replace=TRUE))))
  m <- qgcomp::qgcomp.glm.boot(form,data=d,expnms=c('x1','x2'),q=NULL,
       family=stats::binomial(),rr=TRUE,B=B,seed=seed)
  list(name=name,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=if('z' %in% names(d)) unname(d$z) else NULL,
                   indices=unname(schedule-1L)),
       expected=list(psi=unname(m$psi),se=unname(sqrt(m$var.psi)),
                     ci=unname(m$ci),p=unname(tail(m$pval,1L)),
                     risks=unname(as.numeric(tapply(m$y.expected,m$index,mean))),
                     effects=unname(as.numeric(m$bootsamps[2L,]))))
}
cells <- data.frame(x1=c(0,1,0,1),x2=c(0,0,1,1),successes=c(8,13,12,17))
d1 <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i)
 data.frame(x1=rep(cells$x1[i],24),x2=rep(cells$x2[i],24),
            y=c(rep(1,cells$successes[i]),rep(0,24-cells$successes[i])))))
d2 <- expand.grid(x1=0:1,x2=0:1,rep=1:32,KEEP.OUT.ATTRS=FALSE)
d2$z <- as.numeric(d2$rep %% 3)
d2$y <- as.integer((7*d2$rep+3*d2$x1+5*d2$x2) %% 19 <
                   (5+2*d2$x1+2*d2$x2+d2$z))
cases <- list(make_case('binary_rr_two_exposures',d1,y~x1+x2),
              make_case('binary_rr_adjusted',d2,y~x1+x2+z))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R risk-ratio cases exported\n')
