#!/usr/bin/env Rscript
# Optional direct reference for qgcomp's pscl::zeroinfl NB count engine.
lib <- Sys.getenv("QGCOMP_R_LIB", "../.r-library-qgcomp-2.19.6")
.libPaths(c(lib, .libPaths()))
stopifnot(as.character(packageVersion("pscl")) == "1.5.9")

i <- 0:719
x1 <- i %% 4
x2 <- (i %/% 4) %% 4
z <- sin(i*.17)
mu <- exp(.3+.18*x1-.11*x2+.12*z)
pi <- plogis(-.8+.15*x1-.12*x2+.1*z)
u0 <- ((i*37) %% 719+.5)/719
u1 <- ((i*113) %% 719+.5)/719
y <- ifelse(u0 < pi, 0, qnbinom(u1, size=1.2, mu=mu))
fit <- pscl::zeroinfl(y ~ x1+x2+z, data=data.frame(y, x1, x2, z),
                      dist="negbin",
                      control=pscl::zeroinfl.control(maxit=1000, reltol=1e-12,
                                                     EM=FALSE))
emit <- function(name, value) {
  cat(name, paste(format(as.vector(value), digits=17, scientific=TRUE),
                  collapse="\t"), sep="\t")
  cat("\n")
}
emit("n", length(y))
emit("sum_y", sum(y))
emit("theta", fit$theta)
emit("coefficients", coef(fit))
emit("covariance", vcov(fit))
emit("se_logtheta", fit$SE.logtheta)
