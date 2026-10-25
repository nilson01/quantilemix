#!/usr/bin/env Rscript
# Optional direct reference for qgcomp's pscl::hurdle negative-binomial engine.
lib <- Sys.getenv("QGCOMP_R_LIB", "../.r-library-qgcomp-2.19.6")
.libPaths(c(lib, .libPaths()))
stopifnot(as.character(packageVersion("pscl")) == "1.5.9")

i <- 0:479
x1 <- i %% 4
x2 <- (i %/% 4) %% 4
z <- sin(i*.17)
mu <- exp(.3+.15*x1-.1*x2+.2*z)
theta <- 1.3
p <- plogis(.2+.2*x1-.15*x2-.1*z)
u0 <- ((i*37) %% 479+.5)/479
u1 <- ((i*113) %% 479+.5)/479
p0 <- dnbinom(0, size=theta, mu=mu)
y <- ifelse(u0 < p, pmax(1, qnbinom(pmin(p0+u1*(1-p0),
                                          1-.Machine$double.eps), size=theta, mu=mu)), 0)
fit <- pscl::hurdle(y ~ x1+x2+z, data=data.frame(y, x1, x2, z),
                    dist="negbin", zero.dist="binomial")
emit <- function(name, value) {
  cat(name, paste(format(as.vector(value), digits=17, scientific=TRUE),
                  collapse="\t"), sep="\t")
  cat("\n")
}
emit("n", length(y))
emit("sum_y", sum(y))
emit("theta", fit$theta)
emit("count", coef(fit, model="count"))
emit("zero", coef(fit, model="zero"))
emit("count_cov", vcov(fit, model="count"))
emit("zero_cov", vcov(fit, model="zero"))
