# SPDX-License-Identifier: GPL-2.0-or-later
# Two fixed Bernoulli-logit cases from isolated installed R qgcomp 2.19.6.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript export_binomial_r.R LIBRARY OUTPUT_JSON")
lib <- normalizePath(args[[1L]], mustWork = TRUE)
out <- args[[2L]]
out_link <- Sys.readlink(out)
if (file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop("OUTPUT_JSON already exists")
.libPaths(c(lib, normalizePath(.Library, mustWork = TRUE)), include.site = FALSE)
if (as.character(utils::packageVersion("qgcomp", lib.loc = lib)) != "2.19.6") {
  stop("installed qgcomp 2.19.6 is required")
}
if (!requireNamespace("qgcomp", lib.loc = lib, quietly = TRUE) ||
    !requireNamespace("jsonlite", lib.loc = lib, quietly = TRUE)) {
  stop("qgcomp/jsonlite or dependencies are missing")
}
if (!identical(normalizePath(getNamespaceInfo(asNamespace("qgcomp"), "path")),
               normalizePath(file.path(lib, "qgcomp")))) {
  stop("loaded qgcomp namespace is outside the requested library")
}

make_case <- function(name, d, formula, q, new) {
  model <- qgcomp::qgcomp.glm.noboot(formula, data = d,
      expnms = c("x1", "x2"), q = q, family = stats::binomial())
  summary_fit <- summary(model$fit)
  scored <- if (is.null(q)) d else qgcomp::quantize(d, c("x1", "x2"), q=q)$data
  new_scored <- if (is.null(q)) new else
      qgcomp::quantize(new, c("x1", "x2"), q=q, breaks=model$breaks)$data
  list(
    name=name,
    inputs=list(y=unname(d$y), x1=unname(d$x1), x2=unname(d$x2),
                z=if ("z" %in% names(d)) unname(d$z) else NULL,
                q=q, new_x1=unname(new$x1), new_x2=unname(new$x2),
                new_z=if ("z" %in% names(new)) unname(new$z) else NULL),
    expected=list(
      coefficient_names=names(stats::coef(model$fit)),
      coefficients=unname(stats::coef(model$fit)),
      covariance=unname(summary_fit$cov.scaled),
      scores=unname(as.matrix(scored[, c("x1", "x2")])),
      breaks=model$breaks,
      psi=unname(model$psi), variance=unname(model$var.psi),
      se=unname(sqrt(model$var.psi)), ci=unname(model$ci),
      zstat=unname(tail(model$zstat,1L)),
      p=unname(tail(model$pval,1L)), residual_df=model$fit$df.residual,
      fitted=unname(stats::fitted(model$fit)),
      predictions=unname(stats::predict(model$fit,newdata=new_scored,type="response")),
      iterations=model$fit$iter
    )
  )
}

cells <- data.frame(x1=c(0,1,0,1), x2=c(0,0,1,1), successes=c(5,10,10,15))
d1 <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
  data.frame(x1=rep(cells$x1[i],20),x2=rep(cells$x2[i],20),
             y=c(rep(1,cells$successes[i]),rep(0,20-cells$successes[i])))
}))
new1 <- data.frame(x1=c(0,1,0,1),x2=c(0,0,1,1))
d2 <- expand.grid(x1=0:3,x2=0:3,rep=1:8,KEEP.OUT.ATTRS=FALSE)
d2$z <- as.numeric(d2$rep %% 3)
d2$y <- as.integer((2*d2$x1+3*d2$x2+5*d2$rep) %% 11 < (4+d2$x1))
new2 <- data.frame(x1=c(0,1,3),x2=c(1,2,3),z=c(0,1,2))
cases <- list(make_case("binary_exact_scores",d1,y~x1+x2,NULL,new1),
              make_case("binary_quantized_adjusted",d2,y~x1+x2+z,4L,new2))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion("qgcomp")),
                jsonlite_version=as.character(utils::packageVersion("jsonlite")),
                qgcomp_origin=normalizePath(getNamespaceInfo(asNamespace("qgcomp"),"path")),
                library=lib,cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null="null")
cat(length(cases),"installed-R binomial cases exported\n")
