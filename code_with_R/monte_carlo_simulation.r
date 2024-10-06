# num_sims = 10000
# spot = 100
# maturity = 1
# r = 0.02
# q = 0
# vol = 0.2
# frequency = 'daily'

library(ggplot2)
library(reshape2)

MC.GBM.Equity <- function(num_sims,spot,maturity,r,q,vol,frequency=c("annual", "semi-annual",
                                                                   "quarterly", "monthly", "daily")){
  frequency <- match.arg(frequency)
  dt <- switch(frequency, annual = 1, `semi-annual` = 0.5,
                  quarterly = 0.25, monthly = 1/12, weekly = 1/52, daily = 1/252)
  num_steps = maturity/dt

  if(round(maturity %% dt, digits = 5)!=0){stop("Maturity should be a multiple of the frequency!")}

  set.seed(123)
  epsilon <- matrix(rnorm(num_steps*num_sims), ncol = num_sims, nrow = num_steps)
  gbm <- exp((r - q - ((vol * vol)/2)) * dt + vol * epsilon * sqrt(dt))
  gbm <- apply(rbind(rep(spot, num_sims), gbm), 2, cumprod)

  return(gbm)
}

Execution.Time <- function(num_sims,spot,maturity,r,q,vol,frequency=c("annual", "semi-annual",
                                                                       "quarterly", "monthly", "daily")){
  num_sims <- c(100, 1000, 10000, 100000, 1000000)
  df <- data.frame(num_simulations = numeric(), elapsed_time = numeric())
  for (num_sim in num_sims){
    elapsed_time <- system.time(MC.GBM.Equity(num_sim, spot, maturity, r, q, vol, frequency))["elapsed"]
    df <- rbind(df, data.frame(num_simulations = num_sim, elapsed_time = elapsed_time))
  }
  return(df)
}

# cl <- parallel::makeForkCluster()
# doParallel::registerDoParallel(cl)
# S_mat <- data.frame(foreach(k = 1:num_sim, .combine='rbind') %dopar% {
#   set.seed(k)
#   GBM.Equity(spot, r, q, vol, maturity, num_steps)
# })
# parallel::stopCluster(cl)
#
# colnames(S_mat) = 0:num_steps
#
# return(S_mat)

MC.Option_Price <- function(gbm, K, r, maturity, type="C"){
  maturity_prices <- gbm[nrow(gbm), ]
  if (type=="C"){
    fair_price <- sum(pmax(maturity_prices-K, 0))/ncol(gbm)
  }
  else{
    fair_price <- sum(pmax(K-maturity_prices, 0))/ncol(gbm)
  }
  price <- exp(-r*maturity)*fair_price
  return(price)
}

Display.Prices <- function(gbm, maturity){
  time_steps <- seq(0, maturity, length.out=nrow(gbm))
  gbm_df <- data.frame(time = time_steps, gbm)
  gbm_melt <- melt(gbm_df, id.vars = "time") 
  
  ggplot(gbm_melt, aes(x = time, y = value, group = variable, color = variable)) +
    geom_line() +
    labs(title = "Prices Paths", x = "Time", y = "Prices") +
    theme_minimal()
}