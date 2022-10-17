# ############################################################################################################
#
# AUXILIARY FUNCTIONS - IC_glmnet
#
# ############################################################################################################

IC_glmnet = function(X, y, ic = c("aic", "bic", "aicc", "hq"), alpha, ...){
  
  T = length(y)
  model = glmnet(x = X, y = y, alpha = alpha, ...)
  coefs = coef(model)
  lambdas = model$lambda
  df = model$df
  
  # Ridge regression degrees of freedom
  if (alpha == 0) {
    
    X_std = as.matrix(scale(X))
    
    for (i in 1:length(lambdas)){
      
      chol_aux = chol2inv(chol(t(X_std) %*% X_std + lambdas[i]*diag(ncol(X_std))))
      H_lambda = X_std %*% chol_aux %*% t(X_std)
      df[i] = sum(diag(H_lambda))
      
    }
    
  }
  
  # Elastic Net degrees of freedom
  if (alpha > 0 & alpha < 1) {
    
    X_std = as.matrix(scale(X))
    
    for (i in 1:length(lambdas)){
      aux = coefs[, i]
      aux_notZero = aux[aux != 0]
      vars_selected = names(aux_notZero)[-1]
      
      if (length(vars_selected) == 0){
        
        df[i] = 0
        
      } else {
        
        X_A = as.matrix(X_std[, vars_selected])
        H_A = X_A %*% solve(t(X_A) %*% X_A + (lambdas[i]*(1-alpha))*diag(ncol(X_A))) %*% t(X_A)
        df[i] = sum(diag(H_A))
        
      }
    }
  }
  
  yhat = as.matrix(cbind(1, X)) %*% coefs
  resid = (y - yhat)
  mse = colMeans(resid^2)
  nVar = df + 1
  
  aic = T*log(mse) + 2*nVar
  bic = log(mse) + nVar*log(T)/T
  aicc = aic + (2*nVar*(nVar + 1))/(T - nVar - 1)
  hq = log(mse) + nVar*log(log(T))/T
  
  crit = switch(ic, aic = aic, bic = bic, aicc = aicc, hq = hq)
  selecIndex = which(crit == min(crit))
  
  bestLambda = lambdas[selecIndex]
  bestIC = crit[selecIndex]
  bestnVar = nVar[selecIndex]
  bestCoefs = coefs[, selecIndex]
  bestFitted = yhat[, selecIndex]
  bestResid = resid[, selecIndex]
  
  result = list(lambda = bestLambda, ic = bestIC, nVar = bestnVar, coefficients = bestCoefs,
                fitted = bestFitted, residuals = bestResid)
  
  return(result)
  
}