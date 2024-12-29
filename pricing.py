import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import options as opt


### Rajouter les dividendes !!!

### Computation of the Greeks
def get_delta(function, epsilon=1E-6, **kwargs):
    if 'S0' not in kwargs:
        raise ValueError("The 'S0' parameter (spot price) is required in **kwargs.")
    
    kwargs_plus = kwargs.copy()
    kwargs_plus['S0'] += epsilon

    kwargs_minus = kwargs.copy()
    kwargs_minus['S0'] -= epsilon

    price_plus = function(**kwargs_plus)
    price_minus = function(**kwargs_minus)
    return (price_plus-price_minus)/(2*epsilon)

def get_gamma(function, epsilon=1.8E-2, **kwargs):
    if 'S0' not in kwargs:
        raise ValueError("The 'S0' parameter (spot price) is required in **kwargs.")
    
    kwargs_plus = kwargs.copy()
    kwargs_plus['S0'] += epsilon

    kwargs_minus = kwargs.copy()
    kwargs_minus['S0'] -= epsilon

    price_plus = function(**kwargs_plus)
    price_minus = function(**kwargs_minus)
    price = function(**kwargs)
    return (price_plus+price_minus-2*price)/(epsilon**2)

def get_rho(function, epsilon=1E-6, **kwargs):
    if 'r' not in kwargs:
        raise ValueError("The 'r' parameter (interest rate in percent) is required in **kwargs.")
    
    kwargs_plus = kwargs.copy()
    kwargs_plus['r'] += epsilon

    kwargs_minus = kwargs.copy()
    kwargs_minus['r'] -= epsilon

    r_plus = function(**kwargs_plus)
    r_minus = function(**kwargs_minus)
    return (r_plus-r_minus)/(2*epsilon)*0.01 #for 1% change in interest rate

def get_vega(function, epsilon=1E-6, **kwargs):
    if 'vol' not in kwargs:
        raise ValueError("The 'vol' parameter (volatility) is required in **kwargs.")
    
    kwargs_plus = kwargs.copy()
    kwargs_plus['vol'] += epsilon

    kwargs_minus = kwargs.copy()
    kwargs_minus['vol'] -= epsilon

    vol_plus = function(**kwargs_plus)
    vol_minus = function(**kwargs_minus)
    return (vol_plus-vol_minus)/(2*epsilon)*0.01 #for 1% change in volatility

def get_theta(function, delta_t=0.1/365, **kwargs):
    if 'T' not in kwargs:
        raise ValueError("The 'T' parameter (time to maturity) is required in **kwargs.")
    
    kwargs_plus = kwargs.copy()
    kwargs_plus['T'] += delta_t

    time_current = function(**kwargs)
    time_plus = function(**kwargs_plus)

    return ((time_current-time_plus)/delta_t)/365  # percentage change for 1 day

def get_greeks(function, epsilon=1E-6, **kwargs):
    
    greeks_function = {
        'delta': get_delta,
        'gamma': get_gamma,
        'vega': get_vega,
        'rho': get_rho,
        'theta': get_theta
    }

    greeks_df = pd.DataFrame(list(greeks_function.items()), columns=['Greek', 'Function'])

    def compute_greek(row):
        greek_name = row['Greek']
        greek_function = row['Function']

        if greek_name == 'gamma':
            epsilon_adjusted = epsilon / 5E-5
        else:
            epsilon_adjusted = epsilon

        try:
            result = greek_function(function, epsilon_adjusted, **kwargs)
            return result
        
        except Exception as e:
            print(f"Error computing the {greek_name}: {e}")
            return None

    greeks_df['Value'] = greeks_df.apply(compute_greek, axis=1)

    return greeks_df[['Greek', 'Value']]

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

    # Fix random seed for reproductibility
    np.random.seed(123)

    N=int(round(T*252))   ## 252 times for each trading day in a year
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
    N=int(round(T*252))   ## 252 times for each trading day in a year
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
    barrier : The barrier value that needs to be reached for the option to pay out (float)
    T : maturity (in years)
    r : risk-free rate
    vol : implied volatility
    digit : value to be earned if the barrier is reached
    M : number of simulations
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

### Price a digital strip using Monte Carlo Simulations

def digital_strip_price(S0, barriers, maturities, r, vol, digit, memory, M, antithetic=False):
    """
    S0 : spot price at time t=0
    barriers : list of barriers for each digit option
    maturities : list of maturities (in years) for each digit option
    r : risk-free rate
    vol : implied volatility
    digit : value to be earned if the barrier of the digit is reached
    M : number of simulations
    antithetic : Boolean type (True to improve precision of results)
    """

    # Get all asset prices for each simulation path
    asset_prices=monte_carlo_vectorized(S0, max(maturities), r, vol, M, antithetic)

    num_steps=[int(round(maturity*252)) for maturity in maturities]

    # Extract the final prices for each digit option
    final_prices = asset_prices.iloc[num_steps, :].reset_index(drop=True)

    # Create DataFrames for barriers and times for vectorized operations
    barriers_df = pd.DataFrame(np.tile(barriers, (final_prices.shape[1], 1)).T, index=final_prices.index, columns=final_prices.columns)
    maturities_df = pd.DataFrame(np.tile(maturities, (final_prices.shape[1], 1)).T, index=final_prices.index, columns=final_prices.columns)

    # Initialize matrices for calculations
    memo_df = pd.DataFrame(np.zeros((len(maturities), final_prices.shape[1])), index=final_prices.index, columns=final_prices.columns)
    PV_digits_df = pd.DataFrame(np.zeros((len(maturities), final_prices.shape[1])), index=final_prices.index, columns=final_prices.columns)

    # Calculate the payoff for each simulation path
    PV_digits_df = np.where(final_prices >= barriers_df, 
                            (1 + memo_df) * np.exp(-r * maturities_df) * digit, 0)
    
    # Update memory matrix
    memo_df = np.where(final_prices >= barriers_df, 
                       0, memory * (memo_df + 1))

    # Compute the option price for all digits
    PV_digits_df = pd.DataFrame(PV_digits_df, index=final_prices.index, columns=final_prices.columns)
    digital_strip_price = PV_digits_df.sum(axis=0).mean()

    return digital_strip_price

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
    S0 : spot price at time t=0
    K : strike price
    T1 : time for the beginning of the observations (observation period : T2 - T1)
    T2 : maturity (in years)
    n : number of observations (during the period T2-T1)
    r : risk-free rate
    vol : implied volatility
    M : number of simulations
    antithetic : Boolean type (True to improve precision of results)
    type : 'arithmetic' or 'geometric'

    Return the price of an Asian option (for either a an "arithmetic" or "geometric" option)
    """

    # Get all asset prices for each simulation path
    asset_prices=monte_carlo_vectorized(S0, T2, r, vol, M, antithetic)
    asset_prices=asset_prices[round(T1*252):]

    # Get the prices for the different observations
    observations=[int(x) for x in list(np.linspace(0, len(asset_prices)-1, n+1))][1:]
    observed_prices=asset_prices.iloc[observations, :]

    # Compute average price for the different observations
    if type=='arithmetic':
        average_price=observed_prices.sum(axis=0)/n

    elif type=='geometric':
        average_price=observed_prices.prod(axis=0)**(1/n)
    
    else:
        raise ValueError("type should be either 'arithmetic' or 'geometric'")

    # Compute the payoff actualized
    payoff=np.exp(-r*T2)*np.maximum(average_price-K, 0).mean()

    return payoff

### Price an autocallable

def autocallable_price(A, S0, H1, H2, H3, c, coupon_period, T, r, vol, M, antithetic=False, conditionnal=False):
    """
    c : value of the coupon
    coupon_period : period for paying the coupon (in months)
    conditionnal : Boolean type
    """
    # Number of observations
    n=round(T/(coupon_period/12))   # Convert coupon_period in years and get the number of observations
    alpha=round(A/S0)

    asset_prices=monte_carlo_vectorized(S0, T, r, vol, M, antithetic)
    final_prices=asset_prices.iloc[-1, :]


    observations=[int(x) for x in list(np.linspace(0, len(asset_prices)-1, n+1))][1:]
    observed_prices=asset_prices.iloc[observations, :]

    knock_out_barrier=pd.DataFrame(np.where(observed_prices>H3, True, False), index=observed_prices.index, columns=observed_prices.columns)
    knock_out_barrier_full = pd.DataFrame(False, index=asset_prices.index, columns=asset_prices.columns)
    knock_out_barrier_full.iloc[knock_out_barrier.index] = knock_out_barrier
    knock_out_barrier_full=knock_out_barrier_full.cumsum(axis=0).astype(bool)
    knock_out_barrier=knock_out_barrier_full.iloc[observed_prices.index, :]

    first_knock_out_indices = knock_out_barrier.idxmax(axis=0)
    redeem_payoff=pd.DataFrame(np.where(first_knock_out_indices!=0, np.exp(-r*first_knock_out_indices*(1/252))*A, np.exp(-r*T)*A), index=knock_out_barrier.columns).T

    if conditionnal:
        coupons=pd.DataFrame(np.where((observed_prices>H2) & (knock_out_barrier==False), A*c, 0), index=observed_prices.index, columns=observed_prices.columns)
    else:
        coupons=pd.DataFrame(np.where(knock_out_barrier==False, A*c, 0), index=observed_prices.index, columns=observed_prices.columns)
    
    discount_factors = np.exp(-r * (observed_prices.index / 252))
    coupons = coupons.multiply(discount_factors, axis=0)

    knock_in_barrier=pd.DataFrame(np.where((asset_prices<H1)&(knock_out_barrier_full==False), True, False), index=asset_prices.index, columns=asset_prices.columns)
    knock_in_barrier=knock_in_barrier.cumsum(axis=0).astype(bool)
    knock_in_barrier=pd.DataFrame(np.where(asset_prices<H1, True, False), index=asset_prices.index, columns=asset_prices.columns).any(axis=0)
    option_payoff=pd.DataFrame(np.where(knock_in_barrier==True, -alpha*np.exp(-r*T)*np.maximum(S0-final_prices, 0), 0), index=asset_prices.columns).T

    payoff=pd.concat([coupons, option_payoff, redeem_payoff], axis=0, ignore_index=True).sum(axis=0)
    price=payoff.mean()

    return price