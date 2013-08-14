##' Compute linear mixed model deviance
##'
##' A pure \code{R} implementation of Doug Bates'
##' penalized least squares (PLS) approach for computing
##' linear mixed model deviances. The purpose
##' is to clarify how PLS works without having
##' to read through C++ code, and as a sandbox for
##' trying out modifications to PLS.
##'
##' @param theta covariance parameters
##' @param lmod output of \code{lFormula}
##' @param y response
##' @param weights prior weights
##' @param offset offset
##' @param REML calculate REML deviance?
##'
##' @return deviance with a bunch of attributes:
##' \itemize{
##' \item[devML] ML deviance
##' \item[devREML] REML deviance
##' \item[beta] fixed effects at convergence
##' \item[u] conditional mode at convergence
##' \item[sigmaML] ML estimate of the residual standard deviation 
##' \item[sigmaREML] REML estimate of the residual standard deviation
##' }
##' @export
pls <- function(theta, lmod, y,
                weights = numeric(0),
                offset = numeric(0),
                REML = TRUE){
                                        # expand lmod
    Lind <- lmod$reTrms$Lind
    Lambdat <- lmod$reTrms$Lambdat
    Zt <- lmod$reTrms$Zt
    X <- lmod$X
    n <- nrow(X)
    p <- ncol(X)
    q <- nrow(Zt)
                                        # update Lambdat
    Lambdat@x <- theta[Lind]
                                        # calculate weighted design matrices
    if (length(weights)) {
        W <- Diagonal(x = sqrt(weights))
        Ut <- Zt%*%W
        V <- W%*%X
        wtResp <- as.vector(W %*% y)
    } else {
        Ut <- Zt
        V <- X
        wtResp <- y
    }
    LamtUt <- Lambdat%*%Ut
    ## calculate decomposition of cross-product matrix
    L <- Cholesky(tcrossprod(LamtUt), LDL = FALSE, Imult=1)
    RZX <- solve(L, LamtUt%*%V, system = "L")
    RX <- try(crossprod(V) - crossprod(RZX))
    if(inherits(RX, "try-error")) return(NA)
    ## solve for fixed effect estimates and conditional modes
    cu <- solve(L, LamtUt%*%wtResp, system = "L")
    cb <- solve(t(RX), crossprod(V,wtResp) - crossprod(RZX,cu))
    beta <- solve(RX, cb)
    u <- solve(L, cu - RZX%*%beta, system = "Lt")
    ## calculate conditional mean of the response
    mu <- crossprod(Lambdat%*%Zt,u) + (X%*%beta)
    if (length(offset)) mu <- mu + offset
                                        # calculate weighted residuals
    wtres <- y - mu
    if (length(weights)) wtres <- W %*% wtres
                                        # calculate residual sums of squares
    wrss <- sum(wtres^2)
    pwrss <- wrss + sum(u^2)            # penalize
                                        # calculate log determinants
    ldL2 <- determinant(L, logarithm=TRUE)
    ldRX2 <- 2*determinant(RX, logarithm=TRUE)
    attributes(ldL2) <- attributes(ldRX2) <- NULL
    
                                        # calculate profiled deviance for theta
    devML <- ldL2 + n*(1 + log(2*pi*pwrss) - log(n))
    devREML <- ldL2 + ldRX2 + (n-p)*(1 + log(2*pi*pwrss) -
                                     log(n-p))

    if(REML) dev <- devREML
    else dev <- devML

    sigmaML <- sqrt(pwrss/sum(weights))
    
    return(structure(dev, devML = devML, devREML = devREML, beta = beta, u = u,
                     sigmaML = sigmaML,
                     sigmaREML = sigmaML*(n/(n-p))))
}
