import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import options_trading as opt


### Get M Monte Carlo simulations of asset prices over the time period
def monte_carlo_iteratif(S0, T, r, vol, N, M):
    dt=T/N
    nudt=(r-0.5*vol**2)*dt
    volsdt=vol*np.sqrt(dt)
    erdt=np.exp(-r*dt)

    S=pd.DataFrame()

    for i in range(M):
        St=pd.Series(index=np.arange(0,N,1)).rename(f"{i+1}")
        St[0]=S0
    
        for j in range (1, N+1):
            epsilon=np.random.normal()
            Stn=St[j-1]*np.exp(nudt+volsdt*epsilon)
            St[j]=Stn

        S=pd.concat([S, St], axis=1)

    return S

def monte_carlo_vectorized(S0, T, r, vol, N, M):
    dt=T/N
    nudt=(r-0.5*vol**2)*dt
    volsdt=vol*np.sqrt(dt)
    erdt=np.exp(-r*dt)
    epsilon=np.random.normal(0, 1, (N, M))
    growth_factors=np.exp(nudt + volsdt * epsilon)
    prices=S0*pd.DataFrame(growth_factors).cumprod(axis=0)
    S=pd.concat([pd.DataFrame([S0] * M).T, prices], ignore_index=True)
    S.columns=[f"{i+1}" for i in range(M)]

    return S

### Heston Model
def heston_model(S0, v0, r, rho, kappa, theta, sigma, T, N, M):
    dt = T / N
    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])

    S = np.full(shape=(N + 1, M), fill_value=S0)  # Initialisation des prix à S0
    v = np.full(shape=(N + 1, M), fill_value=v0)  # Initialisation des variances à v0

    # Générer des échantillons de variables aléatoires multivariées
    Z = np.random.multivariate_normal(mu, cov, (N, M))

    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp((r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i] = np.maximum(v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)

    return S, v

### Price a vanilla option with Monte Carlo

def vanilla_option_iteratif(S0, K, T, r, vol, N, M):
    dt=T/N
    nudt=(r-0.5*vol**2)*dt
    volsdt=vol*np.sqrt(dt)
    erdt=np.exp(-r*dt)

    sum_CT=0
    sum_CT2=0

    for i in range(M):
        St=S0
    
        for j in range (N):
            epsilon=np.random.normal()
            Stn=St*np.exp(nudt+volsdt*epsilon)
            St=Stn
        
        CT=max(0,St-K)
            
        sum_CT=sum_CT+CT
        sum_CT2=sum_CT2+CT*CT
    
    C0=np.exp(-r*T)*sum_CT/M
    sigma=np.sqrt((sum_CT2-sum_CT*sum_CT/M)*np.exp(-2*r*T)/(M-1))
    SE=sigma/np.sqrt(M)    #Compute the standard error
    
    return (C0, SE)

def vanilla_option(S0, K, T, r, vol, N, M, opt_type='C'):
    """
    opt_type : "C" for a call and "P" for a put
    """
    asset_prices=monte_carlo_vectorized(S0, T, r, vol, N, M)
    final_prices=asset_prices.iloc[-1, :]
    
    if opt_type=='C':
        payoffs=np.maximum(final_prices - K, 0)
    elif opt_type=='P':
        payoffs=np.maximum(K-final_prices, 0)
    else:
        print("opt_type should be either 'C' for a call option or 'P' for a put option")

    option_price=np.exp(-r*T)*payoffs.mean()

    return option_price


### Price a digit option using Monte Carlo Simulations

def digit_option(S0, barrier, T, r, vol, digit, N, M):
    asset_prices=monte_carlo_vectorized(S0, T, r, vol, N, M)
    final_prices=asset_prices.iloc[-1, :]

    payoffs=np.where(final_prices>barrier, digit, 0)

    digit_price=np.exp(-r*T)*payoffs.mean()

    return digit_price