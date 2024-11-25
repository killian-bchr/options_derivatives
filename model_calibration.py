import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import options as opt
from scipy.optimize import least_squares
from scipy.interpolate import griddata

#FRED API : 54dc21b344d021c79ae6bcd13013fd62

fred = Fred(api_key='54dc21b344d021c79ae6bcd13013fd62')


# Get risk-free rate with zero rates bonds
def get_riskfree_rate():

    # Load all data available in Fred API : zero rate for these bonds
    data_3m = fred.get_series('DGS3MO', observation_start='2024-01-01', observation_end=datetime.today())
    data_6m = fred.get_series('DGS6MO', observation_start='2024-01-01', observation_end=datetime.today())
    data_1y = fred.get_series('DGS1', observation_start='2024-01-01', observation_end=datetime.today())
    data_2y = fred.get_series('DGS2', observation_start='2024-01-01', observation_end=datetime.today())
    data_3y = fred.get_series('DGS3', observation_start='2024-01-01', observation_end=datetime.today())
    data_5y = fred.get_series('DGS5', observation_start='2024-01-01', observation_end=datetime.today())
    data_7y = fred.get_series('DGS7', observation_start='2024-01-01', observation_end=datetime.today())
    data_10y = fred.get_series('DGS10', observation_start='2024-01-01', observation_end=datetime.today())

    # Concatenation of these data
    dataset = pd.concat([data_3m, data_6m, data_1y, data_2y, data_3y, data_5y, data_7y, data_10y], axis=1)
    dataset.columns = ['3', '6', '12', '24', '36', '60', '84', '120']
    dataset.dropna(inplace=True)

    # Re-arrange with new empty columns corresponding to the other months
    all_months = [str(month) for month in range(1, 121)]
    missing_columns = [month for month in all_months if month not in dataset.columns]
    missing_df = pd.DataFrame(np.nan, index=dataset.index, columns=missing_columns)

    dataset = pd.concat([dataset, missing_df], axis=1)
    dataset=dataset[all_months]

    # Interpolate
    dataset = dataset.interpolate(method='linear', axis=1)
    riskfree_rate = dataset.iloc[-1, :]
    #riskfree_rate.dropna(axis=0, inplace=True)

    return riskfree_rate

def plot_riskfree_rate():
    dataset=get_riskfree_rate()
    plt.plot(np.linspace(0, 10, len(dataset)), dataset)

# Get the implied volatility
def implied_vol_newton(S, K, T, r, option_price, vol_init=0.3, epsilon=0.00001):
    """Calculate the implied volatility of an European option with the Newton-Raphson method"""
    vol_new=None
    for i in range (500):
        bs_price=opt.black_scholes(r, S, K, T, vol_init)
        vega=opt.vega_calc(r, S, K, T, vol_init)*100   ##we don't want the vega for 1% change in volatility
        if vega !=0:
            vol_new=vol_init-(bs_price-option_price)/vega
        else:
            break
        new_bs_price=opt.black_scholes(r, S, K, T, vol_new)
        
        if (abs(vol_new-vol_init)<epsilon or abs(new_bs_price-option_price)<epsilon):
            break
        vol_init=vol_new
        
        if vol_new is None or np.isnan(vol_new) or np.isinf(vol_new):
            raise ValueError("Implied volatility calculation did not converge or resulted in invalid value")
            
        if vol_new > 1000:
            raise ValueError("Implied volatility is too large, calculation may be diverging")
        
    return vol_new*100 if isinstance(vol_new, (int, float)) else None

def implied_vol(S, K, T, r, option_price, vol_init=0.3, epsilon=0.00001):
    """Calculate the implied volatility of an European option with a scipy optimizer"""
    def diff(sigma):
        if sigma==0:
            return np.nan
        else:
            bs_price=opt.black_scholes(r, S, K, T, sigma)
        return np.abs(option_price-bs_price)
    implied_vol=(least_squares(diff, [vol_init]).x).item()
    return implied_vol*100


##Get the volatility surface
def volatility_surface(ticker, r=0.02, price=False, method="optimizer"):
    """
    method : "optimizer" or "newton"
    """
    df=yf.download(ticker, start='2024-06-01')
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    #today=datetime.now().date()
    #S=df.loc[today, 'Close']
    S=df['Close'].iloc[-1]
    exp_dates=opt.get_options_dates(ticker)
    time_to_maturity=[]
    strike=[]
    implied_volatility=[]
    for date in exp_dates:
        strikes=opt.call(ticker, date)['strike']
        option_prices=opt.call(ticker, date)['lastPrice']
        days_to_maturity=(datetime.strptime(date, '%Y-%m-%d').date()-datetime.now().date()).days
        maturity=days_to_maturity/365
        for i in range (len(strikes)):
            time_to_maturity.append(maturity)
            strike.append(strikes.iloc[i])
            if maturity>0:
                if method=="optimizer":
                    implied_volatility.append(implied_vol(S, strikes.iloc[i], maturity, r, option_prices.iloc[i]))
                elif method=="newton":
                    implied_volatility.append(implied_vol_newton(S, strikes.iloc[i], maturity, r, option_prices.iloc[i]))
                else:
                    print("method should be either 'optimizer' or 'newton'")
            else:
                implied_volatility.append(np.nan)    ##l'option a expiré donc il n'y a pas de volatilité implicite
        
    data=pd.DataFrame({'Time to maturity':time_to_maturity, 'Strikes':strike, 'Implied volatility':implied_volatility})
    if price:
        return data, S
    else:
        return data

def plot_volatility_surface(ticker, r=0.02, method="optimizer"):
    data, S=volatility_surface(ticker, r, price=True)
    surface=data.pivot_table(values='Implied volatility', index='Strikes', columns='Time to maturity').dropna()

    fig=plt.figure(figsize=(12,6))

    ax=fig.add_subplot(111, projection='3d')
    x, y, z = surface.columns.values, surface.index.values, surface.values

    X, Y =np.meshgrid(x,y)

    ax.set_xlabel("Days to expiration")
    ax.set_ylabel("Strike price")
    ax.set_zlabel("Implied volatility (%)")
    ax.set_title("Volaility Surface")

    surf = ax.plot_surface(X, Y, z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ##Show the spot price of the underlying
    #X_mesh, Z_mesh = np.meshgrid(x, y)
    #Y_stock_price = np.full_like(X_mesh, S)
    #ax.plot_surface(X_mesh, Y_stock_price, Z_mesh, color='red', alpha=0.3)

    plt.show()


# Get rikfree-rate and implied volatility
def calibration(ticker, K, T):
    S0=yf.download(ticker).iloc[-1]['Adj Close']

    # Get the riskfree-rate
    rates=get_riskfree_rate()
    r=rates[str(T*12)]/100

    # Get the volatility surface
    volatility_surface_=volatility_surface(ticker, r=r)

    maturities=volatility_surface_['Time to maturity'].values
    strikes=volatility_surface_['Strikes'].values
    volatilities=volatility_surface_['Implied volatility'].values

    # Interpolate if necessary to get the implied volatility
    points=np.array([maturities, strikes]).T
    IV=griddata(points, volatilities, (T, K), method='linear').item()/100

    return r, IV