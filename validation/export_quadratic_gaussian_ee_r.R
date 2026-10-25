# SPDX-License-Identifier: GPL-2.0-or-later
# Fixed degree-two Gaussian estimating-equation cases from installed qgcomp.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript export_quadratic_gaussian_ee_r.R LIBRARY OUTPUT_JSON')
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

grid <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
grid$id <- ceiling(seq_len(nrow(grid))/4)
grid$z <- as.numeric((grid$rep+grid$x1)%%3)
grid$w <- as.numeric(1+(seq_len(nrow(grid))*7+grid$id)%%4)
grid$x1 <- grid$x1+0.017*grid$rep+0.001*seq_len(nrow(grid))
grid$x2 <- grid$x2+0.013*grid$rep+0.0007*rev(seq_len(nrow(grid)))
grid$y <- 1+0.4*grid$x1-0.2*grid$x2+0.12*grid$x1^2+
          0.08*grid$x2^2+0.06*grid$x1*grid$x2+0.5*grid$z+
          0.09*grid$x1*grid$z-0.04*grid$x2*grid$z+
          0.12*((grid$id*7)%%5-2)+0.03*((seq_len(nrow(grid))*11)%%13-6)

make_case <- function(name, formula, pairwise, covariate, weighted){
  if(weighted){
    m <- qgcomp::qgcomp.glm.ee(formula,data=grid,expnms=c('x1','x2'),
        q=4L,id='id',weights=w,family=stats::gaussian(),degree=2L,
        seed=137L,verbose=FALSE)
  } else {
    m <- qgcomp::qgcomp.glm.ee(formula,data=grid,expnms=c('x1','x2'),
        q=4L,id='id',family=stats::gaussian(),degree=2L,
        seed=137L,verbose=FALSE)
  }
  levels <- unname(sort(unique(m$index)))
  standardized <- vapply(levels,function(a){
    selected <- m$index==a
    if(weighted) stats::weighted.mean(m$y.expected[selected],grid$w)
    else mean(m$y.expected[selected])
  },numeric(1L))
  # Independent fresh-design calculation: each intervention recomputes the
  # formula terms before cluster score aggregation. The installed method uses
  # model.frame subsets in its cluster meat, which retain precomputed I(...) terms.
  d <- grid
  d$x1 <- m$qx$x1_q
  d$x2 <- m$qx$x2_q
  beta <- unname(m$fit$est)
  theta <- unname(m$msmfit$est)
  scores <- lapply(unique(d$id),function(group){
    rows <- d[d$id==group,,drop=FALSE]
    wg <- if(weighted) rows$w else rep(1,nrow(rows))
    X <- model.matrix(formula,rows)
    Xi <- do.call(rbind,lapply(levels,function(a){
      altered <- rows
      altered$x1 <- a
      altered$x2 <- a
      model.matrix(formula,altered)
    }))
    Mi <- do.call(rbind,lapply(levels,function(a){
      cbind(1,rep(a,nrow(rows)),rep(a*a,nrow(rows)))
    }))
    c(as.vector(t(X)%*%(wg*(rows$y-X%*%beta))),
      as.vector(t(Mi)%*%(rep(wg,times=length(levels))*(Xi%*%beta-Mi%*%theta))))
  })
  fresh_meat <- Reduce('+',lapply(scores,function(s) outer(s,s)))
  inverse_bread <- solve(m$bread)
  fresh_covariance <- inverse_bread%*%fresh_meat%*%t(inverse_bread)
  # Rebuild the installed helper's saved-model-frame path separately. This
  # confirms which stale derived columns cause the reported meat difference.
  stale_frame <- model.frame(formula,d)
  stale_scores <- lapply(unique(d$id),function(group){
    rows <- stale_frame[d$id==group,,drop=FALSE]
    wg <- if(weighted) d$w[d$id==group] else rep(1,nrow(rows))
    X <- model.matrix(formula,rows)
    Xi <- do.call(rbind,lapply(levels,function(a){
      altered <- rows
      altered$x1 <- a
      altered$x2 <- a
      model.matrix(formula,altered)
    }))
    Mi <- do.call(rbind,lapply(levels,function(a){
      cbind(1,rep(a,nrow(rows)),rep(a*a,nrow(rows)))
    }))
    c(as.vector(t(X)%*%(wg*(rows$y-X%*%beta))),
      as.vector(t(Mi)%*%(rep(wg,times=length(levels))*(Xi%*%beta-Mi%*%theta))))
  })
  stale_rebuilt_meat <- Reduce('+',lapply(stale_scores,function(s) outer(s,s)))
  if(max(abs(stale_rebuilt_meat-m$meat))>1e-7) stop('stale model-frame meat did not match R result')
  list(name=name,pairwise=pairwise,covariate_interactions=covariate,
       weighted=weighted,
       inputs=list(y=unname(grid$y),x1=unname(grid$x1),x2=unname(grid$x2),
                   z=unname(grid$z),id=unname(grid$id),
                   w=if(weighted) unname(grid$w) else NULL,q=4L),
       expected=list(conditional_names=names(m$fit$est),
                     conditional_coefficients=unname(m$fit$est),
                     conditional_covariance=unname(m$fit$vcov),
                     msm_coefficients=unname(m$msmfit$est),
                     msm_covariance=unname(m$msmfit$vcov),
                     full_covariance=unname(m$covmat.all_robust),
                     bread=unname(m$bread),meat=unname(m$meat),
                     stale_rebuilt_meat=unname(stale_rebuilt_meat),
                     fresh_meat=unname(fresh_meat),
                     fresh_full_covariance=unname(fresh_covariance),
                     psi=unname(m$psi),variance=unname(m$var.psi),
                     ci=unname(m$ci),statistic=unname(m$tstat),
                     p=unname(m$pval),df=unname(m$df),
                     scores=unname(as.matrix(m$qx)),breaks=m$breaks,
                     levels=levels,standardized=unname(standardized)))
}
square <- y~x1+x2+I(x1^2)+I(x2^2)+z
pairwise <- y~x1+x2+I(x1^2)+I(x2^2)+I(x1*x2)+z
covariate <- y~x1+x2+I(x1^2)+I(x2^2)+I(x1*x2)+I(x1*z)+I(x2*z)+z
cases <- list(make_case('quadratic_square_ee',square,FALSE,FALSE,FALSE),
              make_case('quadratic_pairwise_ee',pairwise,TRUE,FALSE,FALSE),
              make_case('quadratic_covariate_weighted_ee',covariate,TRUE,TRUE,TRUE))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion('qgcomp')),
                jsonlite_version=as.character(utils::packageVersion('jsonlite')),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null='null')
cat(length(cases),'installed-R quadratic Gaussian EE cases exported\n')
