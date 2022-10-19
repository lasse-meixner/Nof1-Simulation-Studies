library(nlme)


ar1_model = function(
  y0,y1,sub,T,t,c0
){
  y0 <- as.vector(y0)
  y1 <- as.vector(y1)
  sub <- as.vector(sub)
  t <- as.vector(t)
  c0 <- as.vector(c0)
  m0 <- nlme::lme(y0 ~ T + t + c0, random = ~ 1 | sub, correlation = corAR1( form = ~ 1 | sub))
  m1 <- nlme::lme(y1 ~ T + t + c0, random = ~ 1 | sub, correlation = corAR1( form = ~ 1 | sub))
  p0_value <- summary(fit)$tTable[2,5]
  t1_value <- summary(fit)$tTable[2,4]
  fe1_value <- summary(fit)$tTable[2,1]
  return(c(p0_value,t1_value,fe1_value))
}