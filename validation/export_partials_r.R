# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed, disjoint-score reference for installed qgcomp 2.19.6.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('Usage: Rscript export_partials_r.R LIB OUTPUT_JSON')
lib <- normalizePath(args[[1L]], mustWork=TRUE)
out <- args[[2L]]
if (file.exists(out)) stop('output already exists')
.libPaths(c(lib, normalizePath(.Library,mustWork=TRUE)), include.site=FALSE)
if (as.character(utils::packageVersion('qgcomp',lib.loc=lib)) != '2.19.6') stop('qgcomp 2.19.6 required')
if (!requireNamespace('qgcomp',lib.loc=lib,quietly=TRUE) ||
    !requireNamespace('jsonlite',lib.loc=lib,quietly=TRUE)) stop('missing dependency')
if (!identical(normalizePath(getNamespaceInfo(asNamespace('qgcomp'),'path')),
               normalizePath(file.path(lib,'qgcomp')))) stop('wrong qgcomp origin')
library(qgcomp, lib.loc=lib)
d <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
d$z <- as.numeric(d$rep %% 3)
d$y <- 1 + 0.8*d$x1 - 0.45*d$x2 + 0.2*d$z + 0.1*sin(seq_len(nrow(d)))
train <- d[d$rep <= 4,,drop=FALSE]
valid <- d[d$rep > 4,,drop=FALSE]
res <- qgcomp::qgcomp.partials(fun='qgcomp.glm.noboot',
    f=y~x1+x2+z, q=NULL, expnms=c('x1','x2'),
    traindata=train,validdata=valid,family=stats::gaussian())
fields <- function(fit) {
  if (is.null(fit)) return(NULL)
  list(psi=unname(fit$psi),se=sqrt(unname(fit$var.psi)),
       ci=unname(fit$ci),p=unname(tail(fit$pval,1L)))
}
set.seed(19)
clustered <- data.frame(id=rep(1:3,each=4),value=seq_len(12))
cluster_split <- qgcomp::split_data(clustered,cluster='id',prop.train=0.5)
payload <- list(version=as.character(utils::packageVersion('qgcomp')),
    train=list(y=unname(train$y),x1=unname(train$x1),x2=unname(train$x2),z=unname(train$z)),
    valid=list(y=unname(valid$y),x1=unname(valid$x1),x2=unname(valid$x2),z=unname(valid$z)),
    positive=unname(res$posmix),negative=unname(res$negmix),
    training=fields(res$train.fit),positive_fit=fields(res$pos.fit),
    negative_fit=fields(res$neg.fit),
    cluster_split=list(trainidx=unname(cluster_split$trainidx),
                       valididx=unname(cluster_split$valididx),
                       overlapping_ids=unname(intersect(unique(cluster_split$traindata$id),
                                                        unique(cluster_split$validdata$id)))))
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
