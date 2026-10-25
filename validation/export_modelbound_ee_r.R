# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed Gaussian EE curve bounds from an isolated installed qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_modelbound_ee_r.R LIBRARY OUTPUT_JSON')
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
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:6,KEEP.OUT.ATTRS=FALSE)
grid$id <- ceiling(seq_len(nrow(grid))/3)
grid$z <- as.numeric((grid$rep+grid$x1)%%3)
grid$x1 <- grid$x1+0.017*grid$rep+0.001*seq_len(nrow(grid))
grid$x2 <- grid$x2+0.013*grid$rep+0.0007*rev(seq_len(nrow(grid)))
grid$y <- 1+0.4*grid$x1-0.2*grid$x2+0.5*grid$z+
          0.12*((grid$id*7)%%5-2)+0.03*((seq_len(nrow(grid))*11)%%13-6)
d <- grid[,c('y','x1','x2','z','id')]
m <- qgcomp::qgcomp.glm.ee(y~x1+x2+z,data=d,expnms=c('x1','x2'),
      q=4L,id='id',family=stats::gaussian(),degree=1,seed=137L,verbose=FALSE)
one <- function(pwonly) {
  set.seed(418L)
  tab <- qgcomp::modelbound.ee(m,alpha=0.05,pwonly=pwonly)
  set.seed(418L)
  draws <- qgcomp:::.rmvnorm(5000L,stats::coef(m),stats::vcov(m))
  list(pwonly=pwonly,columns=names(tab),table=unname(lapply(tab,unname)),draws=unname(draws))
}
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                            z=unname(d$z),id=unname(d$id),q=4L),
                coefficients=unname(stats::coef(m)),
                covariance=unname(stats::vcov(m)),
                cases=list(one(FALSE),one(TRUE)))
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat('Two installed-R Gaussian EE bound tables exported\n')
