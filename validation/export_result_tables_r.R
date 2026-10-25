# SPDX-License-Identifier: GPL-2.0-or-later
# Compare public result tables using fixed inputs from the existing R fixtures.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3L) stop("Usage: Rscript export_result_tables_r.R LIBRARY FIXTURES_DIR OUTPUT_JSON")
lib <- normalizePath(args[[1L]],mustWork=TRUE)
fixtures <- normalizePath(args[[2L]],mustWork=TRUE)
out <- args[[3L]]
out_link <- Sys.readlink(out)
if(file.exists(out) || (!is.na(out_link) && nzchar(out_link))) stop("OUTPUT_JSON already exists")
.libPaths(c(lib,normalizePath(.Library,mustWork=TRUE)),include.site=FALSE)
if(as.character(utils::packageVersion("qgcomp",lib.loc=lib))!="2.19.6") stop("qgcomp 2.19.6 required")
for(pkg in c("qgcomp","jsonlite","generics","broom")) {
  if(!requireNamespace(pkg,lib.loc=lib,quietly=TRUE)) stop(paste("missing",pkg))
  origin <- normalizePath(getNamespaceInfo(asNamespace(pkg),"path"))
  if(!identical(origin,normalizePath(file.path(lib,pkg)))) stop(paste("wrong",pkg,"origin"))
}

read_case <- function(file,name) {
  payload <- jsonlite::fromJSON(file.path(fixtures,file),simplifyVector=FALSE)
  hit <- which(vapply(payload$cases,function(x) identical(x$name,name),logical(1)))
  if(length(hit)!=1L) stop(paste("fixture case missing:",name))
  payload$cases[[hit]]$inputs
}
as_rows <- function(frame) lapply(seq_len(nrow(frame)),function(i) as.list(frame[i,,drop=FALSE]))
make_tables <- function(name,model,with_glance=TRUE,exponentiate=FALSE) {
  raw <- generics::tidy(model)
  exp <- if(exponentiate) as_rows(generics::tidy(model,exponentiate=TRUE)) else NULL
  list(name=name,tidy=as_rows(raw),tidy_quick=as_rows(generics::tidy(model,quick=TRUE)),
       tidy_exponentiated=exp,
       glance=if(with_glance) as.list(generics::glance(model)[1,,drop=FALSE]) else NULL)
}
make_glm <- function(file,name,family) {
  a <- read_case(file,name)
  if(!is.null(a$exposures)) {
    x <- as.data.frame(do.call(rbind,lapply(a$exposures,unlist)))
    expnms <- paste0("x",seq_len(ncol(x)))
    names(x) <- expnms
    d <- data.frame(y=unlist(a$y),x)
    if(!is.null(a$covariates)) {
      z <- as.data.frame(do.call(rbind,lapply(a$covariates,unlist)))
      names(z) <- paste0("z",seq_len(ncol(z)))
      d <- cbind(d,z)
    }
  } else {
    expnms <- c("x1","x2")
    d <- data.frame(y=unlist(a$y),x1=unlist(a$x1),x2=unlist(a$x2))
    if(!is.null(a$z)) d$z <- unlist(a$z)
  }
  rhs <- paste(names(d)[names(d)!="y"],collapse="+")
  model <- qgcomp::qgcomp.glm.noboot(
    stats::as.formula(paste("y~",rhs)),data=d,expnms=expnms,
    q=a$q,family=family,alpha=if(is.null(a$alpha)) 0.05 else a$alpha)
  make_tables(name,model,exponentiate=!identical(family$family,"gaussian"))
}
make_cox <- function(name) {
  a <- read_case("r_cox_2_19_6.json",name)
  d <- data.frame(time=unlist(a$time),event=unlist(a$event),
                  x1=unlist(a$x1),x2=unlist(a$x2),z=unlist(a$z))
  model <- qgcomp::qgcomp.cox.noboot(
    survival::Surv(time,event)~x1+x2+z,data=d,expnms=c("x1","x2"),q=a$q)
  make_tables(name,model,with_glance=FALSE,exponentiate=TRUE)
}
cases <- list(
  make_glm("r_gaussian_2_19_6.json","adjusted_mixed",stats::gaussian()),
  make_glm("r_gaussian_2_19_6.json","prequantized",stats::gaussian()),
  make_glm("r_binomial_2_19_6.json","binary_exact_scores",stats::binomial()),
  make_glm("r_binomial_2_19_6.json","binary_quantized_adjusted",stats::binomial()),
  make_glm("r_poisson_2_19_6.json","count_exact_scores",stats::poisson()),
  make_glm("r_poisson_2_19_6.json","count_quantized_adjusted",stats::poisson()),
  make_cox("cox_quantized_untied"))
payload <- list(R_version=R.version.string,
                qgcomp_version=as.character(utils::packageVersion("qgcomp")),
                jsonlite_version=as.character(utils::packageVersion("jsonlite")),
                broom_version=as.character(utils::packageVersion("broom")),
                cases=cases)
jsonlite::write_json(payload,out,pretty=TRUE,auto_unbox=TRUE,digits=NA,null="null")
cat(length(cases),"installed-R result-table cases exported\n")
