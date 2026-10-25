# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed clustered Gaussian estimating-equation cases from installed qgcomp.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_gaussian_ee_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) || !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
make_case <- function(name,d,q){
  m <- qgcomp::qgcomp.glm.ee(y~x1+x2+z, data=d,
      expnms=c('x1','x2'), q=q, id='id', family=stats::gaussian(),
      degree=1, seed=137L, verbose=FALSE)
  list(name=name,
       inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=unname(d$z),id=unname(d$id),q=q),
       expected=list(conditional_coefficients=unname(m$fit$est),
                     conditional_covariance=unname(m$fit$vcov),
                     msm_coefficients=unname(m$msmfit$est),
                     msm_covariance=unname(m$msmfit$vcov),
                     full_covariance=unname(m$covmat.all_robust),
                     bread=unname(m$bread),meat=unname(m$meat),
                     psi=unname(m$psi),variance=unname(m$var.psi),
                     ci=unname(m$ci),statistic=unname(m$tstat),
                     p=unname(m$pval),df=unname(m$df),
                     scores=unname(as.matrix(m$qx)),breaks=m$breaks,
                     levels=unname(sort(unique(m$index))),
                     means=unname(as.numeric(tapply(m$y.expected,m$index,mean)))))
}
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:6,KEEP.OUT.ATTRS=FALSE)
grid$id <- ceiling(seq_len(nrow(grid))/3)
grid$z <- as.numeric((grid$rep+grid$x1)%%3)
grid$y <- 1+0.4*grid$x1-0.2*grid$x2+0.5*grid$z+
          0.12*((grid$id*7)%%5-2)+0.03*((seq_len(nrow(grid))*11)%%13-6)
d1 <- grid[,c('y','x1','x2','z','id')]
d2 <- grid
d2$x1 <- d2$x1+0.017*d2$rep+0.001*seq_len(nrow(d2))
d2$x2 <- d2$x2+0.013*d2$rep+0.0007*rev(seq_len(nrow(d2)))
d2$y <- 1+0.4*d2$x1-0.2*d2$x2+0.5*d2$z+
        0.12*((d2$id*7)%%5-2)+0.03*((seq_len(nrow(d2))*11)%%13-6)
d2 <- d2[,c('y','x1','x2','z','id')]
cases <- list(make_case('gaussian_ee_prequantized',d1,NULL),
              make_case('gaussian_ee_quantized',d2,4L))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R Gaussian EE cases exported\n')
