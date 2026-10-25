#!/usr/bin/env Rscript
# Optional fixed-case reference from isolated installed qgcomp 2.19.6.
lib <- Sys.getenv("QGCOMP_R_LIB", "../.r-library-qgcomp-2.19.6")
.libPaths(c(lib, .libPaths()))
stopifnot(as.character(packageVersion("qgcomp")) == "2.19.6")
library(qgcomp)
library(jsonlite)

make_case <- function(case) {
  i <- seq_len(240L)
  x1 <- (i * 7L + case) %% 4L
  x2 <- (i * 11L + (i %/% 4L) + case) %% 4L
  z <- sin(i * 1.7 + case)
  mu <- exp(0.35 + 0.22*x1 - 0.14*x2 + 0.10*z)
  p_nonzero <- plogis(0.4 + 0.14*x1 - 0.20*x2 + 0.15*z)
  u0 <- ((i * 109L + case*23L) %% 239L + 0.5)/240
  u1 <- ((i * 73L + (i %/% 3L)*17L + case*19L) %% 239L + 0.5)/240
  y <- ifelse(u0 >= p_nonzero, 0L, qpois(exp(-mu) + u1*(1-exp(-mu)), mu))
  data.frame(y=y, x1=x1, x2=x2, z=z)
}

cases <- lapply(1:2, function(case) {
  dat <- make_case(case)
  fit <- qgcomp.hurdle.noboot(y ~ x1 + x2 + z | x1 + x2 + z,
                               data=dat, expnms=c("x1", "x2"), q=NULL,
                               dist="poisson", control=pscl::hurdle.control(maxit=1000, reltol=1e-12))
  list(data=dat, psi=fit$psi, se=lapply(fit$var.psi, sqrt),
       coefficients=fit$fit$coefficients,
       covariance=unname(as.matrix(vcov(fit$fit))))
})
write_json(list(reference_version="2.19.6", cases=cases),
           "tests/fixtures/r_hurdle_2_19_6.json", auto_unbox=TRUE, digits=16, pretty=TRUE)
