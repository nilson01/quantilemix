# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed unequal-time marginal Poisson cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop('Usage: Rscript export_poisson_bootstrap_offset_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')

make_case <- function(name,d,form,q){
  B <- 24L
  seed <- 991L
  set.seed(seed)
  schedule <- t(replicate(B,sort(sample(seq_len(nrow(d)),nrow(d),replace=TRUE))))
  m <- qgcomp::qgcomp.glm.boot(form,data=d,expnms=c('x1','x2'),q=q,
       family=stats::poisson(),B=B,seed=seed)
  list(name=name,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=if('z' %in% names(d)) unname(d$z) else NULL,
                   time=unname(d$time),q=q,indices=unname(schedule-1L)),
       expected=list(psi=unname(m$psi),se=unname(sqrt(m$var.psi)),
                     ci=unname(m$ci),p=unname(tail(m$pval,1L)),
                     means=unname(as.numeric(tapply(m$y.expected,m$index,mean))),
                     scores=unname(as.matrix(m$qx)),breaks=m$breaks,
                     effects=unname(as.numeric(m$bootsamps[2L,]))))
}
cells <- data.frame(x1=c(0,1,0,1),x2=c(0,0,1,1),mean=c(1,2,3,6))
d1 <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
  time <- 1:5
  data.frame(x1=rep(cells$x1[i],5),x2=rep(cells$x2[i],5),
             time=time,y=cells$mean[i]*time+c(0,1,-1,2,-2))
}))
d2 <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
d2$z <- as.numeric(d2$rep %% 3)
d2$time <- as.numeric(1+d2$rep %% 5)
d2$y <- as.integer((2*d2$x1+3*d2$x2+5*d2$rep) %% 9)
cases <- list(make_case('offset_bootstrap_exact_scores',d1,y~x1+x2+offset(log(time)),NULL),
              make_case('offset_bootstrap_quantized_adjusted',d2,y~x1+x2+z+offset(log(time)),4L))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                qgcomp_origin=normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R offset Poisson bootstrap cases exported\n')
