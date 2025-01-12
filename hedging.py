import pricing
import model_calibration as model
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize

### Delta Hedging

def delta_hedge_option(function, num_hedging_adjustments, long=True, **kwargs):

    S0 = kwargs.get('S0', 100)
    T = kwargs.get('T', 1)
    r = kwargs.get('r', 0.02)
    vol = kwargs.get('vol', 0.2)
    M = kwargs.get('M', 50000)
    antithetic = kwargs.get('antithetic', False)

    new_kwargs = {key: value for key, value in kwargs.items() if key not in ['S0', 'T']}

    epsilon = 1 if long else -1

    asset_prices = pricing.monte_carlo_vectorized(S0=S0, T=T, r=r, vol=vol, M=M, antithetic=antithetic)
    if num_hedging_adjustments > len(asset_prices):
        raise ValueError("The number of dynamic hedging adjustments must be less than or equal to the number of timesteps in the asset price simulation")
    
    hedging_adjustments = np.linspace(0, len(asset_prices)-1, num_hedging_adjustments).astype(int)
    prices = asset_prices.loc[hedging_adjustments].mean(axis=1)
    df = pd.DataFrame(columns=['Asset Prices'], data=prices)
    df['Option Price'] = df.apply(lambda row: function(S0=row['Asset Prices'], T=T-float(row.name)/252, **new_kwargs), axis=1)
    df['Delta'] = df.apply(lambda row: pricing.get_delta(function, 
            S0=row['Asset Prices'], T=T-float(row.name)/252, **new_kwargs), axis=1)
    df['Shares Value'] = epsilon*df['Asset Prices']*df['Delta']
    df['Cash Position'] = np.where(df.index==0 , -epsilon*df['Delta']*df['Asset Prices']*np.exp(r*(T-df.index/252)), 
            epsilon*(df['Delta'].shift(1)-df['Delta'])*df['Asset Prices']*np.exp(r*(T-df.index/252))).cumsum()
    
    # Compute the payoff of the option hedged
    dictionary=pricing.dictionary()
    if function in dictionary:
        compute_payoff = dictionary[function]
    else:
        raise ValueError("There is no function yet to compute the payoff of this derivative")
    payoff_kwargs = {key: value for key, value in kwargs.items() if key not in ['S0']}
    df['Option Payoff'] = df.apply(lambda row: compute_payoff(S=row['Asset Prices'], long=long, **payoff_kwargs), axis=1)

    df['PnL'] = df['Shares Value'] + df['Cash Position'] - epsilon*df.iloc[0]['Option Price'] + epsilon*df['Option Payoff']
    return df


### Gamma Hedging

def gamma_hedge_option(function, num_hedging_adjustments, long=True, **kwargs):

    # Assumption : for the option to hedge, we use the same underlying asset -> same r, S, vol and T but K different

    S0 = kwargs.get('S0', 100)
    T = kwargs.get('T', 1)
    r = kwargs.get('r', 0.02)
    vol = kwargs.get('vol', 0.2)
    M = kwargs.get('M', 50000)
    antithetic = kwargs.get('antithetic', False)

    new_kwargs = {key: value for key, value in kwargs.items() if key not in ['S0', 'T']}

    epsilon = 1 if long else -1

    asset_prices = pricing.monte_carlo_vectorized(S0=S0, T=T, r=r, vol=vol, M=M, antithetic=antithetic)
    if num_hedging_adjustments > len(asset_prices):
        raise ValueError("The number of dynamic hedging adjustments must be less than or equal to the number of timesteps in the asset price simulation")
    
    hedging_adjustments = np.linspace(0, len(asset_prices)-1, num_hedging_adjustments).astype(int)
    prices = asset_prices.loc[hedging_adjustments].mean(axis=1)
    df = pd.DataFrame(columns=['Asset Prices'], data=prices)
    df['Option Price'] = df.apply(lambda row: function(S0=row['Asset Prices'], T=T-float(row.name)/252, **new_kwargs), axis=1)
    df['Gamma'] = df.apply(lambda row: pricing.get_gamma(function,
            S0=row['Asset Prices'], T=T-float(row.name)/252, **new_kwargs), axis=1)
    
    df['Optimal Strike'] = 0.0
    df['Hedge Option Gamma'] = 0.0
    df['Options To Hedge'] = 0.0
    df['Price Options To Hedge'] = 0.0
    df['Total Gamma'] = 0.0
    df['Delta'] = 0.0
    df['Option Payoff'] = 0.0

    active_options = []

    def diff(params, S, maturity, target):
        K = params
        return (target-pricing.gamma_calc(S0=S, K=K, T=maturity, r=r, vol=vol, opt_type='C'))**2
    
    hedge_payoffs = []

    # Compute the payoffs of the options hedged
    def calculate_payoff(S, active_options):
        payoffs = [option['Quantity'] * max(0, S - option['Strike']) for option in active_options]   ## Assumption : we buy only call for gamma hedge
        return sum(payoffs)
    
    # Compute the total delta
    def calculate_total_delta(S, active_options):
        total_delta = sum(
            option['Quantity'] * pricing.delta_calc(S0=S, K=option['Strike'], T=maturity, r=r, vol=vol, opt_type='C')
            for option in active_options
        )
        return total_delta

    j=0
    for i in df.index:
        S = df.loc[i, 'Asset Prices']
        maturity = T - float(i) / 252

        gamma_target = df['Gamma'][i] + df['Total Gamma'][j] if i > 0 else df['Gamma'][i]

        if gamma_target>0.05:
            initial_guess = [S0]
            result = minimize(diff, initial_guess, args=(S, maturity, gamma_target), method='L-BFGS-B', bounds=[(0.1, 2 * S0)])
            optimal_strike = result.x[0]
        
            df.loc[i, 'Optimal Strike'] = optimal_strike
            hedge_gamma = -epsilon * pricing.gamma_calc(S0=S, K=optimal_strike, T=maturity, r=r, vol=vol, opt_type='C')
            df.loc[i, 'Hedge Option Gamma'] = hedge_gamma

            if hedge_gamma != 0:
                options_to_hedge = -gamma_target / hedge_gamma
                df.loc[i, 'Options To Hedge'] = options_to_hedge
                option_price = abs(options_to_hedge) * pricing.black_scholes(S0=S, K=optimal_strike, T=maturity, r=r, vol=vol, opt_type='C')
                df.loc[i, 'Price Options To Hedge'] = option_price

                df.loc[i, 'Total Gamma'] = df['Total Gamma'][j] + options_to_hedge * hedge_gamma if i > 0 else options_to_hedge * hedge_gamma
            
                active_options.append({
                    'Strike': optimal_strike,
                    'Quantity': abs(options_to_hedge),
                })
            else:
                df.loc[i, 'Total Gamma'] = df['Total Gamma'][j] if i > 0 else 0

        else:
            df.loc[i, 'Optimal Strike'] = np.nan
            df.loc[i, 'Hedge Option Gamma'] = np.nan
            df.loc[i, 'Options To Hedge'] = np.nan
            df.loc[i, 'Price Options To Hedge'] = np.nan
            df.loc[i, 'Total Gamma'] = df['Total Gamma'][j]
        
        hedge_payoffs.append(calculate_payoff(df.loc[i, 'Asset Prices'], active_options))

        total_delta_options = calculate_total_delta(S, active_options)
        df.loc[i, 'Delta'] = (
            pricing.get_delta(function, S0=S, T=maturity, **new_kwargs) + total_delta_options
        )

        # Compute the payoff of the option hedged
        dictionary=pricing.dictionary()
        if function in dictionary:
            compute_payoff = dictionary[function]
        else:
            raise ValueError("There is no function yet to compute the payoff of this derivative")
        payoff_kwargs = {key: value for key, value in kwargs.items() if key not in ['S0']}
        df.loc[i, 'Option Payoff'] = compute_payoff(S=df['Asset Prices'][i], long=long, **payoff_kwargs)

        j=i

    df['Shares Value'] = epsilon*df['Asset Prices']*df['Delta']
    df['Cash Position'] = np.where(df.index==0 , -epsilon*df['Delta']*df['Asset Prices']*np.exp(r*(T-df.index/252)), 
            epsilon*(df['Delta'].shift(1)-df['Delta'])*df['Asset Prices']*np.exp(r*(T-df.index/252))).cumsum()
    
    df['Options Buyed To Hedge'] = (df['Options To Hedge'].fillna(0)*df['Price Options To Hedge'].fillna(0))
    df['Options Buyed To Hedge'] = df['Options Buyed To Hedge'].cumsum()

    df['Hedge Option Payoff'] = hedge_payoffs
    
    df['PnL'] = df['Shares Value'] + df['Cash Position'] - epsilon*df.iloc[0]['Option Price'] + epsilon*df['Option Payoff'] + df['Options Buyed To Hedge'] + df['Hedge Option Payoff']
    
    df.fillna(0, inplace=True)
    
    return df[['Asset Prices', 'Option Payoff', 'Option Price', 'Gamma', 'Total Gamma', 'Hedge Option Gamma', 'Options To Hedge', 
               'Price Options To Hedge', 'Options Buyed To Hedge', 'Hedge Option Payoff', 'Delta', 'Shares Value', 'Cash Position', 'PnL']]