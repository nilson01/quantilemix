# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed conditional multinomial cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_multinomial_r.R LIBRARY OUTPUT_JSON')
lib <- normalizePath(args[[1L]],mustWork=TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop('OUTPUT_JSON already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion('qgcomp',lib.loc=lib))!='2.19.6') stop('qgcomp 2.19.6 required')
if(!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE) ||
   !requireNamespace('nnet',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if(!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
              normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
extract <- function(m){
  fit <- m$fit
  joint <- qgcomp::joint_test(m)
  homogeneous <- qgcomp::homogeneity_test(m)
  list(coefficients=unname(stats::coef(fit)),
                     covariance=unname(stats::vcov(fit)),
                     psi=unname(m$psi),psi_covariance=unname(m$covmat.psi),
                     ci=unname(m$ci.psi),
                     p=unname(2*stats::pnorm(abs(m$psi/sqrt(diag(m$covmat.psi))),
                                               lower.tail=FALSE)),
                     fitted_probabilities=unname(stats::predict(fit,type='probs')),
                     scores=unname(as.matrix(m$qx)),breaks=m$breaks,
                     joint_test=list(chisq=unname(joint$chisq),df=unname(joint$df),
                                     p=unname(joint$pvalue)),
                     homogeneity_test=list(chisq=unname(homogeneous$chisq),
                                           df=unname(homogeneous$df),
                                           p=unname(homogeneous$pvalue)))
}
make_case <- function(name,d,q){
  m <- qgcomp::qgcomp.multinomial.noboot(y~x1+x2+z,data=d,
         expnms=c('x1','x2'),q=q,alpha=0.05)
  tight <- qgcomp::qgcomp.multinomial.noboot(y~x1+x2+z,data=d,
         expnms=c('x1','x2'),q=q,alpha=0.05,reltol=1e-13,maxit=1000)
  list(name=name,
       inputs=list(y=as.character(d$y),x1=unname(d$x1),x2=unname(d$x2),
                   z=unname(d$z),q=q,levels=levels(d$y)),
       expected=extract(m),tighter_expected=extract(tight))
}
grid <- expand.grid(x1=0:3,x2=0:3,rep=1:15,KEEP.OUT.ATTRS=FALSE)
i <- seq_len(nrow(grid))
grid$z <- as.numeric(grid$rep %% 3)
categorize <- function(x1,x2,z,u){
  e1 <- exp(-0.5+0.25*x1-0.1*x2+0.3*z)
  e2 <- exp(-0.8-0.15*x1+0.35*x2-0.2*z)
  p0 <- 1/(1+e1+e2)
  p1 <- e1*p0
  factor(ifelse(u<p0,'low',ifelse(u<p0+p1,'mid','high')),
         levels=c('low','mid','high'))
}
d1 <- grid
d1$y <- categorize(d1$x1,d1$x2,d1$z,((i*97)%%241+0.5)/241)
d1 <- d1[,c('y','x1','x2','z')]
d2 <- grid
d2$x1 <- d2$x1+0.011*d2$rep+0.0001*i
d2$x2 <- d2$x2+0.009*d2$rep+0.00005*i
d2$y <- categorize(grid$x1,grid$x2,grid$z,((i*113)%%241+0.5)/241)
d2 <- d2[,c('y','x1','x2','z')]
cases <- list(make_case('multinomial_prequantized',d1,NULL),
              make_case('multinomial_quantized',d2,4L))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                nnet_version=as.character(utils::packageVersion('nnet')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R multinomial cases exported\n')
