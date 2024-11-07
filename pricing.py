import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import options_trading as opt


### Get M Monte Carlo simulations of asset prices over the time period
def monte_carlo_iteratif(S0, T, r, vol, M):
    """
    S0 : spot price at time t=0
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    M : number of simulations
    """
    N=round(T*252)
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

def monte_carlo_vectorized(S0, T, r, vol, M, antithetic=False):
    """
    S0 : spot price at time t=0
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    M : number of simulations
    antithetic : Boolean type (True to improve precision of results)

    Return a matrix with M columns for each simulation where a simulation represents a path for the asset price
    """

    N=round(T*252)   ## 252 times for each trading day in a year
    dt=T/N
    nudt=(r-0.5*vol**2)*dt
    volsdt=vol*np.sqrt(dt)

    if antithetic:
        M=int(round(M/2))

    epsilon=np.random.normal(0, 1, (N, M))
    growth_factors=np.exp(nudt + volsdt * epsilon)
    prices=S0*pd.DataFrame(growth_factors).cumprod(axis=0)

    if antithetic:
        growth_factors_ant=np.exp(nudt - volsdt * epsilon)
        prices_ant=S0*pd.DataFrame(growth_factors_ant).cumprod(axis=0)
        prices=pd.concat([prices, prices_ant], axis=1, ignore_index=True)

    S=pd.concat([pd.DataFrame([S0] * prices.shape[1]).T, prices], ignore_index=True)
    S.columns=[f"{i+1}" for i in range(prices.shape[1])]

    return S

### Heston Model
def heston_model(S0, v0, r, rho, kappa, theta, sigma, T, M):
    N=round(T*252)   ## 252 times for each trading day in a year
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

def vanilla_option_iteratif(S0, K, T, r, vol, M):
    N=round(T*252)
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

def vanilla_option_price(S0, K, T, r, vol, M, opt_type='C', antithetic=False):
    """
    S0 : spot price at time t=0
    K : strike price
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    M : number of simulations
    opt_type : "C" for a call and "P" for a put
    antithetic : Boolean type (True to improve precision of results)

    Return the price of a vanilla option (for either a call option or a put option)
    """

    # Get all asset prices for each simulation path
    asset_prices=monte_carlo_vectorized(S0, T, r, vol, M, antithetic)
    final_prices=asset_prices.iloc[-1, :]
    
    # Compute the payoff depending on if it is a Call or a Put
    if opt_type=='C':
        payoffs=np.maximum(final_prices - K, 0)
    elif opt_type=='P':
        payoffs=np.maximum(K-final_prices, 0)
    else:
        print("opt_type should be either 'C' for a call option or 'P' for a put option")

    # Compute the price actualized
    option_price=np.exp(-r*T)*payoffs.mean()

    return option_price


### Price a digit option using Monte Carlo Simulations

def digit_option_price(S0, barrier, T, r, vol, digit, M, antithetic=False):
    """
    S0 : spot price at time t=0
    barrier : value to reach to get the digit
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    digit : value to be earned is the barrier is reached
    M : number of simulations
    opt_type : "C" for a call and "P" for a put
    antithetic : Boolean type (True to improve precision of results)

    Return the price of a digit option
    """

    # Get all asset prices for each simulation path
    asset_prices=monte_carlo_vectorized(S0, T, r, vol, M, antithetic)
    final_prices=asset_prices.iloc[-1, :]

    # Compute the payoff and assigned a digit if the barrier has been crossed
    payoffs=np.where(final_prices>barrier, digit, 0)

    # Compute the price actualized
    digit_price=np.exp(-r*T)*payoffs.mean()

    return digit_price

### Price a Barrier option using Monte Carlo Simulations

def barrier_option_price(S0, barrier, K, T, r, vol, M, opt_type="C", barrier_type="knock_in", antithetic=False):
    """
    S0 : spot price at time t=0
    barrier : value to knock-in or knock-out the option
    K : strike price
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    M : number of simulations
    opt_type : "C" for a Call and "P" for a Put
    barrier_type : "knock_in" or "knock_out"
    antithetic : Boolean type (True to improve precision of results)

    Return the price of a barrier option (for either a call option or a put option)
    Barrier can be 'knock-in' : option becomes active when the barrier is crossed 
               or 'knock-out' : option becomes inactive when the barrier is crossed
    """

    # Get all asset prices for each simulation path
    asset_prices = monte_carlo_vectorized(S0, T, r, vol, M, antithetic)
    final_prices = asset_prices.iloc[-1, :]

    # Return a Series a Boolean : True if barrier has been crossed and False if not
    crossed_barrier = np.any(asset_prices>barrier, axis=0)

    # Compute the right crossed_barrier Series according to the barrier_type
    if barrier_type == "knock_in":
        active_payoff = crossed_barrier
    elif barrier_type == "knock_out":
        active_payoff = ~crossed_barrier
    else:
        raise ValueError("barrier_type should be either 'knock_in' or 'knock_out'")

    # Compute the payoff depending on if it is a Call or a Put
    if opt_type == "C":
        payoffs = active_payoff * np.maximum(final_prices - K, 0)
    elif opt_type == "P":
        payoffs = active_payoff * np.maximum(K - final_prices, 0)
    else:
        raise ValueError("opt_type should be either 'C' for a call or 'P' for a put")

    # Compute the price actualized
    option_price = np.exp(-r * T) * payoffs.mean()

    return option_price

### Price a Ladder option using Monte Carlo simulations

def ladder_option_price(S0, strikes, barriers, rebates, T, r, vol, M, antithetic=False):
    """
    S0 : spot price at time t=0
    strikes : list of differents strike prices (for each barrier + the final strike)
    barrier : list of differents barriers
    rebates : list of different coupons assigned to the differents barriers
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    M : number of simulations
    antithetic : Boolean type (True to improve precision of results)

    Return the price of a ladder option : an option which pays a coupon for each barrier crossed throughout the life of the option
    """

    # Get all asset prices for each simulation path
    asset_prices=monte_carlo_vectorized(S0, T, r, vol, M, antithetic)
    final_prices=asset_prices.iloc[-1, :]

    payoff = pd.Series(0, index=asset_prices.columns) 

    # Compute the payoff for each barrier
    for i in range(len(barriers)):
        crossed_barrier=np.any(asset_prices>barriers[i], axis=0)

        payoff+=crossed_barrier*np.maximum(barriers[i]-strikes[i], 0)*rebates[i]
    
    # Compute the final payoff
    payoff+=np.maximum(final_prices-strikes[-1], 0)
    
    # Compute the price actualized
    option_price=np.exp(-r*T)*payoff.mean()

    return option_price

### Price a Lookback option using Monte Carlo simulations

def lookback_option_price(S0, K, T, r, vol, M, opt_type='C', strike_type='fixed', antithetic=False):
    """
    S0 : spot price at time t=0
    K : strike price
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    M : number of simulations
    opt_type : "C" for a Call option and "P" for a Put option
    strike_type : "fixed" for "float"
    antithetic : Boolean type (True to improve precision of results)

    Return the price of a lookback option (for either a call option or a put option and either fixed or floating strike)
    """

    # Get all asset prices for each simulation path
    asset_prices=monte_carlo_vectorized(S0, T, r, vol, M, antithetic)
    final_prices=asset_prices.iloc[-1, :]

    max_prices=asset_prices.max(axis=0)
    min_prices=asset_prices.min(axis=0)

    # Compute the payoff for a fixed strike
    if strike_type == 'fixed':
        if opt_type == 'C':
            payoff=np.maximum(max_prices-K, 0)
        elif opt_type == 'P':
            payoff=np.maximum(K-min_prices, 0)
        else:
            raise ValueError("opt_type should be either 'C' for a call or 'P' for a put")
    
    # Compute the payoff for a floating strike
    elif strike_type == 'float':
        if opt_type == 'C':
            payoff=np.maximum(final_prices-min_prices, 0)
        elif opt_type == 'P':
            payoff=np.maximum(max_prices-final_prices, 0)
        else:
            raise ValueError("opt_type should be either 'C' for a call or 'P' for a put")
        
    else:
        raise ValueError("strike_type should be either 'fixed' or 'float' ")
    
    # Compute the price actualized
    lookback_option_price=np.exp(-r*T)*payoff.mean()

    return lookback_option_price

### Price an Asian option using Monte Carlo simulations

def asian_option_price(S0, K, T1, T2, n, r, vol, M, antithetic=False, type='arithmetic'):
    """
    type : 'arithmetic' or 'geometric'
    T2-T1 : correspond to our period of observation
    n : nnumber of observations during the period T2-T1
    """
    asset_prices=monte_carlo_vectorized(S0, T2, r, vol, M, antithetic)
    asset_prices=asset_prices[round(T1*252):]

    observations=[int(x) for x in list(np.linspace(0, (T2-T1)*252, n+1))][1:]

    Sum=pd.DataFrame()

    # On parcourt les observations
    for obs in observations:
        Sum=pd.concat([Sum, asset_prices.iloc[obs]], axis=1)

    if type=='arithmetic':
        Sum=Sum.sum(axis=1)/n

    elif type=='geometric':
        Sum=Sum.prod(axis=1)**(1/n)
    
    else:
        raise ValueError("type should be either 'arithmetic' or 'geometric'")

    payoff=np.maximum(Sum-K, 0).mean()

    return payoff