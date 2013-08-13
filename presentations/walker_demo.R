library(lme4)

#################
# Modularization
#################

## basic example
head(sleepstudy)
lmer(Reaction ~ Days + (Days|Subject), sleepstudy)


## 1.  Parse the data and formula:
lmod <- lFormula(Reaction ~ Days + (Days|Subject),
                 sleepstudy)
names(lmod)
names(lmod$reTrms)
### allows user to modify model inputs
### this is kind of like our answer to glm.fit


## 2.  Create the deviance function to be optimized:
(devfun <- do.call(mkLmerDevfun, lmod))
ls(environment(devfun)) # the environment of devfun contains objects required for its evaluation
ls(environment(devfun)$pp)
ls(environment(devfun)$resp)
##' allows user to evaluate the deviance where
##' ever they like
##' (similar to old devFunOnly argument)
##' uses:
##' - explore deviance surface
##' - pass to favourite optimizer or an mcmc sample
##' - add a prior distribution on the covariance
##'   parameters
##' - compose it with another function to add new
##'   parameters
##'   (e.g. parameters for R-side random effects)



## 3.  Optimize the deviance function:
opt <- optimizeLmer(devfun)
opt[1:3]
### allows user to compare with what default lme4
### optimization will do with their modified model


## 4.  Package up the results:
mkMerMod(environment(devfun), opt, lmod$reTrms,
         fr = lmod$fr)
### package up the results into a merMod object
### (often not useful if the model has been
### severely modified)



## Example from Arne Kaldhusdal and Torsten Hothorn
library(mvtnorm)

g <- 2    # number of groups
n <- 100  # number of repetitions
S <- diag(g)  # correlation matrix of groups
S[cbind(2:1, 1:2)] <- 0.75  # group 1 and 2 are strongly correlated
var.group <- 2  # group variance

set.seed(29)

# correlated data:
Y <- rmvnorm(n, mean = c(-2, 2),
             sigma = var.group * S) +
  matrix(rnorm(n*g, 0, 0.1), n, g)
plot(Y)

# group the data and fit a first model:
d <- data.frame(y = as.vector(t(Y)),
                group = factor(rep(1:2, n)), 
                rep = factor(rep(1:n, each = g)))
head(d)
form <- y ~ group - 1 + (group - 1 | rep)
m <- lmer(form, data = d)
m <- lmer(form, data = d, control =
          lmerControl(check.numobs.vs.rankZ = "ignore"))

# but what if we know the underlying correlation structure?
lmod <- lFormula(form, data = d,
                 control =
                 lmerControl(check.numobs.vs.rankZ =
                             "ignore"))

# Alter the random effects design matrix:
Zt <- lmod$reTrms$Zt
R <- t(chol(S))
CF <- kronecker(Diagonal(n), R) 
newZt <- t(crossprod(lmod$reTrms$Zt, CF))
lmod$reTrms$Zt <- newZt # new RE design matrix

# We only need one variance parameter, and no covariance:
lmod$reTrms$theta <- 1
lmod$reTrms$lower <- 0
Lambdat <- lmod$reTrms$Lambdat
newLambdat <- as(Matrix(diag(nrow(newZt)),
                        sparse = TRUE), 'dgCMatrix')
image(Lambdat[1:10,1:10])
image(newLambdat[1:10,1:10])
lmod$reTrms$Lambdat <- newLambdat

# Modify the mapping between theta and Lambdat
lmod$reTrms$Lind <- rep(1, nrow(newZt))

# There's now only a single intercept
lmod$reTrms$cnms <- list(rep = "intercept")

# fit the modified model
(devfun <- do.call(mkLmerDevfun, lmod))
opt <- optimizeLmer(devfun)
(m <- mkMerMod(environment(devfun), opt, lmod$reTrms,
               fr = lmod$fr))

#############################################
# More control afforded by reference classes
#############################################
glmod <- glFormula(
                   cbind(incidence, size - incidence) ~
                   period + (1 | herd),
                   data = cbpp, family = binomial)
devfun <- do.call(mkGlmerDevfun, glmod)

(rho <- environment(devfun))
ls(rho)
pp <- rho$pp
resp <- rho$resp

class(pp)
class(resp)

merPredD$fields()
merPredD$methods()

head(pp$X)
pp$u(1)

pp$updateXwts(resp$sqrtWrkWt())
pp$updateDecomp()
pp$updateRes(resp$wtWrkResp())
pp$solveU()
pp$u(1)
rho$pp$u(1)
environment(devfun)$pp$u(1)

#################
# lme4 in pure R
#################
library(lme4pureR)

# two functions:  pls and pirls
# they just return an approximate deviance with which to optimize
# over theta (or theta and beta)

# lmm ML
lmod <- lFormula(Reaction ~ Days + (Days|Subject),
                 sleepstudy)
optim(c(1, 0, 1), pls, lmod = lmod,
      y = sleepstudy$Reaction)$par
mML <- lmer(Reaction ~ Days + (Days|Subject),
            sleepstudy, REML = FALSE)
getME(mREML, "theta")

# lmm REML
optim(c(1, 0, 1), pls, lmod = lmod,
      y = sleepstudy$Reaction, REML = FALSE)$par
mREML <- lmer(Reaction ~ Days + (Days|Subject),
              sleepstudy)
getME(mML, "theta")


# glmm nAGQ = 0
glmod <- glFormula(cbind(incidence, size - incidence) ~
                   period + (1 | herd),
              data = cbpp, family = binomial)
optim(1, pirls, mu = rep(0.5, nrow(cbpp)),
      eta = rep(0, nrow(cbpp)),
      glmod = glmod, y = with(cbpp, incidence/size),
      weights = cbpp$size, family = binomial)$par
gm0 <- glmer(cbind(incidence, size - incidence) ~
             period + (1 | herd),
              data = cbpp, family = binomial, nAGQ = 0)
getME(gm0, "theta")

# glmm nAGQ = 1
optim(c(1, rep(0, 4)), pirls1,
      mu = rep(0.5, nrow(cbpp)),
      eta = rep(0, nrow(cbpp)),
      glmod = glmod, y = with(cbpp, incidence/size),
      weights = cbpp$size, family = binomial)$par
gm1 <- glmer(cbind(incidence, size - incidence) ~
             period + (1 | herd),
              data = cbpp, family = binomial, nAGQ = 1)
getME(gm1, "theta")
fixef(gm1)

