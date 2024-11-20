require(dsl, quietly = TRUE)
require(broom, quietly = TRUE)
require(texreg, quietly = TRUE)

tidy.dsl <- function(x, conf.int = FALSE, conf.level = 0.95, exponentiate = FALSE, ...) {
  res <- suppressMessages(dsl:::summary.dsl(object = x, ci = conf.level, ...))
  terms <- row.names(res)
  cols <- c("estimate" = "Estimate", "std.error" = "Std. Error", "p.value" = "p value")
  if (conf.int) {
    cols <- c(cols, "conf.low" = "CI Lower", "conf.high" = "CI Upper")
  }
  out <- as.list(res)[cols]
  names(out) <- names(cols)
  out <- as_tibble(as.data.frame(out))
  out <- bind_cols(term = terms, out)
  if (exponentiate)
    out <- broom:::exponentiate(out)
  return(out)
}
setMethod("tidy", signature = className("list", "dsl"), definition = tidy.dsl)
setMethod("tidy", signature = className("dsl", "dsl"), definition = tidy.dsl)

extract.dsl <- function(model,
                        include.r.squared = TRUE,
                        include.sumsquares = TRUE,
                        include.nobs = TRUE,
                        ...) {
  
  s <- suppressMessages(dsl:::summary.dsl(model, ...))
  coefnames <- row.names(s)
  co <- s[, "Estimate"]
  se <- s[, "Std. Error"]
  pval <- s[, "p value"]
  str(model, 1)
  gof <- numeric()
  gof.names <- character()
  gof.decimal <- logical()
  if (FALSE & isTRUE(include.r.squared)) {
    rs <- s$r.squared
    gof <- c(gof, rs)
    gof.names <- c(gof.names, "HPY R$^2$")
    gof.decimal <- c(gof.decimal, TRUE)
  }
  if (isTRUE(include.sumsquares)) {
    gof <- c(gof, mean(model$RMSE))
    gof.names <- c(gof.names, "mean RMSE")
    gof.decimal <- c(gof.decimal, TRUE)
  }
  if (isTRUE(include.nobs)) {
    gof <- c(gof, model$internal$num_data, model$internal$num_expert)
    gof.names <- c(gof.names, "Num. obs.", "Num. labeled")
    gof.decimal <- c(gof.decimal, FALSE, FALSE)
  }
  
  tr <- texreg:::createTexreg(
    coef.names = coefnames,
    coef = co,
    se = se,
    pvalues = pval,
    gof.names = gof.names,
    gof = gof,
    gof.decimal = gof.decimal
  )
  return(tr)
}
setMethod("extract", signature = className("list", "dsl"), definition = extract.dsl)
