ultimate_forecast = function(df, h, eval_start, eval_end, target_var, type_estimation = 'recursive'){
  
  # Function Details --------------------------------------
  
  # Argument df:
  #   - The argument "df" should be a tibble/dataframe where the first colum stores the dates
  #   - The other columns must contain all the relevant variables (y, x1, ..., xN)
  #   - The first date in df is assumed to be the start of the sample
  #   - The last date in df is assumed to be the final point of the evaluation set (df should be filtered)
  #   - The target variable should be named y
  
  # Argument eval_start and eval_end:
  #   - Both should be in 'Date' format
  
  # Argument type_estimation:
  #   - The options are 'rolling' and 'recursive'
  
  # Defining the target variable --------------------------------------
  
  # Getting y with dates (yt)
  yt = df[, c(1, which(colnames(df) == target_var))]
  colnames(yt)[2] = "y"
  
  # Defining Training and Test Sets --------------------------------------
  
  # Number of points to be forecasted
  P = lubridate::interval(eval_start, eval_end) %/% months(1) + 1
  
  # Test set index (Notation: {y_{T+h|T}, y_{T+h+1|T+1}, ... , y_{T+H|T+H-h}})
  H = P + h -1
  T = nrow(df) - H
  testSet_index = (T + h):(T + H)
  
  # True values in the evaluation set
  y_true = yt[testSet_index, ]
  
  # Defining sequence of training sets
  final_index = T:(T+H-h)
  
  # Organizing these Training Sets indexes into a List (recursive or rolling)
  TrainingSet_index = vector(mode = "list", length = H-h+1)

  for (j in 1:length(TrainingSet_index)){
    
    if (type_estimation == "recursive"){
      TrainingSet_index[[j]] = 1:final_index[j]  
    } else {
      TrainingSet_index[[j]] = j:final_index[j]
    }
    
  }
  
  # Initializing vectors of forecasts --------------------------------------
  
  # Autoregressive Direct Forecasting
  AR1_forec = vector(mode = "numeric", length = P)
  AR_AIC_forec = vector(mode = "numeric", length = P)
  AR_BIC_forec = vector(mode = "numeric", length = P)
  
  # ARMA Iterated forecasts
  ARMA_AIC_forec = vector(mode = "numeric", length = P)
  ARMA_BIC_forec = vector(mode = "numeric", length = P)
  ARMA_AICc_forec = vector(mode = "numeric", length = P)
  
  # Statistical Learning methods
  lasso_aic_forec = vector(mode = "numeric", length = P)
  lasso_bic_forec = vector(mode = "numeric", length = P)
  lasso_aicc_forec = vector(mode = "numeric", length = P)
  RF_forec = vector(mode = "numeric", length = P)
  Xgboost_forec = vector(mode = "numeric", length = P)

  # Simple methods
  mean_forec = vector(mode = "numeric", length = P)
  naive_forec = vector(mode = "numeric", length = P)
  
  # Main Forecasting Loop --------------------------------------
  
  for (j in 1:length(TrainingSet_index)){
    
    # Defining the training set index and y_{t+h}
    trainingSet_index_loop = TrainingSet_index[[j]]
    y = yt$y[(trainingSet_index_loop[1]+h):tail(trainingSet_index_loop, 1)]
    
    # Defining the data matrix
    X_matrix = df[trainingSet_index_loop[1]:(tail(trainingSet_index_loop, 1) - h), -1]
    
    # ARMA Forecasts (iterated)
    y_arima = yt$y[trainingSet_index_loop[1]:tail(trainingSet_index_loop, 1)]
    
    ARMA_AIC = auto.arima(y_arima, d = 0, approximation = FALSE, stepwise = FALSE, ic = "aic")
    ARMA_AIC_forec[j] = as.numeric(forecast(ARMA_AIC, h = h)$mean)[h]
    
    ARMA_BIC = auto.arima(y_arima, d = 0, approximation = FALSE, stepwise = FALSE, ic = "bic")
    ARMA_BIC_forec[j] = as.numeric(forecast(ARMA_BIC, h = h)$mean)[h]
    
    ARMA_AIC = auto.arima(y_arima, d = 0, approximation = FALSE, stepwise = FALSE, ic = "aicc")
    ARMA_AICc_forec[j] = as.numeric(forecast(ARMA_AIC, h = h)$mean)[h]
    
    # LASSO forecasts
    lasso_bic = IC_glmnet(X = as.matrix(X_matrix), y = y, ic = 'bic', alpha = 1)
    lasso_bic_coefs = lasso_bic$coefficients
    lasso_bic_forec[j] = as.matrix(cbind(1, df[tail(trainingSet_index_loop, 1), -1])) %*% as.matrix(lasso_bic_coefs)
    
    lasso_aicc = IC_glmnet(X = as.matrix(X_matrix), y = y, ic = 'aicc', alpha = 1)
    lasso_aicc_coefs = lasso_aicc$coefficients
    lasso_aicc_forec[j] = as.matrix(cbind(1, df[tail(trainingSet_index_loop, 1), -1])) %*% as.matrix(lasso_aicc_coefs)
    
    # # Random Forest Forecasts
    # RF = randomForest(x = X_matrix, y = y, ntree = 500)
    # RF_forec[j] = predict(RF, newdata = df[tail(trainingSet_index_loop, 1), ])
    
  }
  
  # Defining Forecast Errors --------------------------------------
  
  # ARMA Models
  error_arma_aic = y_true$y - ARMA_AIC_forec
  RMSE_arma_aic = sqrt(mean(error_arma_aic^2))
  MAE_arma_aic = mean(abs(error_arma_aic))
  
  error_arma_bic = y_true$y - ARMA_BIC_forec
  RMSE_arma_bic = sqrt(mean(error_arma_bic^2))
  MAE_arma_bic = mean(abs(error_arma_bic))
  
  error_arma_aicc = y_true$y - ARMA_AICc_forec
  RMSE_arma_aicc = sqrt(mean(error_arma_aicc^2))
  MAE_arma_aicc = mean(abs(error_arma_aicc))
  
  # LASSO Models
  error_lasso_bic = y_true$y - lasso_bic_forec
  RMSE_lasso_bic = sqrt(mean(error_lasso_bic^2))
  MAE_lasso_bic = mean(abs(error_lasso_bic))
  
  error_lasso_aicc = y_true$y - lasso_aicc_forec
  RMSE_lasso_aicc = sqrt(mean(error_lasso_aicc^2))
  MAE_lasso_aicc = mean(abs(error_lasso_aicc))
  
  # Random Forest Model
  error_RF = y_true$y - RF_forec
  RMSE_RF = sqrt(mean(error_RF^2))
  MAE_RF = mean(abs(error_RF))
  
  # Return
  return(
    list(
      RMSE = c(RMSE_ARMA_AIC = RMSE_arma_aic, RMSE_ARMA_BIC = RMSE_arma_bic, RMSE_ARMA_AICc = RMSE_arma_aicc,
               RMSE_LASSO_BIC = RMSE_lasso_bic, RMSE_LASSO_AICc = RMSE_lasso_aicc),
      MAE = c(MAE_ARMA_AIC = MAE_arma_aic, MAE_ARMA_BIC = MAE_arma_bic, MAE_ARMA_AICc = MAE_arma_aicc,
              MAE_LASSO_BIC = MAE_lasso_bic, MAE_LASSO_AICc = MAE_lasso_aicc)
    )
  )
  
}
  
  
  
