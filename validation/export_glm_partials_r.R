# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed binary/count partial-effect cases from installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('Usage: Rscript export_glm_partials_r.R LIB OUTPUT_JSON')
lib <- normalizePath(args[[1L]], mustWork=TRUE)
out <- args[[2L]]
if (file.exists(out)) stop('output already exists')
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if (as.character(utils::packageVersion('qgcomp',lib.loc=lib)) != '2.19.6') stop('qgcomp 2.19.6 required')
if (!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
    !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if (!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
               normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
library(qgcomp,lib.loc=lib)

base <- expand.grid(x1=0:3,x2=0:3,rep=1:20,KEEP.OUT.ATTRS=FALSE)
base$z <- as.numeric(base$rep %% 3)
n <- nrow(base)
u1 <- ((seq_len(n)*53 + (seq_len(n)%/%7)*31) %% 317 + 0.5)/318
u2 <- ((seq_len(n)*73 + (seq_len(n)%/%5)*17) %% 311 + 0.5)/312

fields <- function(fit) {
  if (is.null(fit)) return(NULL)
  list(psi=unname(fit$psi),se=sqrt(unname(fit$var.psi)),ci=unname(fit$ci))
}
make_case <- function(family,q) {
  d <- base
  if (!is.null(q)) {
    d$x1 <- d$x1 + 0.04*sin(seq_len(n)*0.71)
    d$x2 <- d$x2 + 0.05*cos(seq_len(n)*0.53)
  }
  if (family=='binomial') {
    probability <- stats::plogis(-0.3 + 0.50*base$x1 - 0.40*base$x2 + 0.13*base$z)
    d$y <- as.integer(u1 < probability)
    link <- stats::binomial()
  } else {
    mean <- exp(0.2 + 0.28*base$x1 - 0.20*base$x2 + 0.10*base$z)
    d$y <- stats::qpois(u2,mean)
    link <- stats::poisson()
  }
  train <- d[d$rep <= 10,,drop=FALSE]
  valid <- d[d$rep > 10,,drop=FALSE]
  res <- qgcomp::qgcomp.partials(fun='qgcomp.glm.noboot',
      f=y~x1+x2+z, q=q, expnms=c('x1','x2'),
      traindata=train,validdata=valid,family=link)
  part <- function(x) list(y=unname(x$y),x1=unname(x$x1),
                           x2=unname(x$x2),z=unname(x$z))
  list(name=paste(family,if(is.null(q)) 'scores' else 'raw',sep='_'),
       family=family,q=q,train=part(train),valid=part(valid),
       positive=unname(res$posmix),negative=unname(res$negmix),
       training=fields(res$train.fit),positive_fit=fields(res$pos.fit),
       negative_fit=fields(res$neg.fit))
}
payload <- list(version=as.character(utils::packageVersion('qgcomp')),
                R_version=R.version.string,
                cases=list(make_case('binomial',NULL),make_case('binomial',4L),
                           make_case('poisson',NULL),make_case('poisson',4L)))
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
