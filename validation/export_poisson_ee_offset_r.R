# Fixed unequal-observation-time probe against installed qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_poisson_ee_offset_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link)))
  stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6')
  stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE))
  stop('missing installed dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')

counts <- matrix(c(1,2,2,3,2,3,5,6,4,5,7,8,9,11,13,15),nrow=4,byrow=TRUE)
d <- expand.grid(cell=1:4,id=0:3)
d$x1 <- c(0,1,0,1)[d$cell]
d$x2 <- c(0,0,1,1)[d$cell]
d$y <- counts[cbind(d$cell,d$id+1)]
d$obstime <- rep(c(0.5,1,1.5,2),4)
ordinary <- qgcomp::qgcomp.glm.ee(y~x1+x2,data=d,expnms=c('x1','x2'),
                                   q=NULL,id='id',family=stats::poisson(),
                                   degree=1,seed=137L,verbose=FALSE)
with_offset <- qgcomp::qgcomp.glm.ee(y~x1+x2,data=d,expnms=c('x1','x2'),
                                      q=NULL,id='id',family=stats::poisson(),
                                      degree=1,seed=137L,verbose=FALSE,
                                      offset=log(d$obstime))
result <- list(
  R_version=R.version.string,
  qgcomp_version=as.character(utils::packageVersion('qgcomp')),
  inputs=list(y=unname(d$y),x1=unname(d$x1),x2=unname(d$x2),
              id=unname(d$id),obstime=unname(d$obstime)),
  no_offset=list(psi=unname(ordinary$psi),var_psi=unname(ordinary$var.psi),
                 coefficients=unname(ordinary$fit$est)),
  supplied_offset=list(psi=unname(with_offset$psi),
                       var_psi=unname(with_offset$var.psi),
                       coefficients=unname(with_offset$fit$est)))
jsonlite::write_json(result,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat('installed-R Poisson EE offset probe exported\n')
