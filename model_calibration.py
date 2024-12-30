import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import options as opt
import seaborn as sns
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
def volatility_surface(ticker_symbol, r=0.02):
    ticker=yf.Ticker(ticker_symbol)
    S=ticker.history(period="1d")['Close'].iloc[-1]
    exp_dates=ticker.options
    df=pd.DataFrame({'exp_dates':exp_dates})
    df['exp_dates'] = pd.to_datetime(df['exp_dates'], format='%Y-%m-%d')
    df['maturity'] = (df['exp_dates'] - pd.Timestamp.now()).dt.days / 365
    df['exp_dates']=df['exp_dates'].dt.strftime('%Y-%m-%d')
    def get_calls_by_date(ticker, date):
        return ticker.option_chain(date).calls[['strike', 'lastPrice']]
    
    option_data = []
    for _, row in df.iterrows():
        options = get_calls_by_date(ticker, row['exp_dates'])
        options['maturity'] = row['maturity']
        option_data.append(options)
    
    #option_data = [get_calls_by_date(ticker, date) for date in df['exp_dates']]
    
    result = pd.concat(option_data, ignore_index=True)
    #result['maturity'] = np.concatenate([np.repeat(maturity, len(options)) for maturity, options in zip(df['maturity'], option_data)])
    
    result['implied vol'] = result.apply(lambda row: implied_vol(S, row['strike'], row['maturity'], r, row['lastPrice']), axis=1)
    result['vega'] = result.apply(lambda row: 100*opt.vega_calc(r=r, S=S, K=row['strike'], T=row['maturity'], sigma=row['implied vol']), axis=1)

    return result[['maturity', 'strike', 'implied vol', 'vega']]

def plot_volatility_surface(ticker, r=0.02):
    data = volatility_surface(ticker, r=r)
    surface = data.pivot_table(values='implied vol', index='strike', columns='maturity').dropna()

    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(111, projection='3d')
    x, y, z = surface.columns.values, surface.index.values, surface.values

    X, Y = np.meshgrid(x,y)

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

def plot_volatility_heatmap(ticker, r=0.02):
    data = volatility_surface(ticker, r=r)
    
    surface = data.pivot_table(values='implied vol', index='strike', columns='maturity').dropna()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        surface,
        annot=False,  # Si vous voulez afficher les valeurs, mettez True
        cmap='viridis',
        cbar_kws={'label': 'Implied Volatility (%)'},  # Légende pour la barre de couleurs
        linewidths=0.5  # Ajoutez des lignes pour délimiter les cases
    )
    
    plt.title("Volatility Heatmap", fontsize=16)
    plt.xlabel("Days to Expiration")
    plt.ylabel("Strike Price")
    plt.xticks(rotation=45)  # Rotation des ticks pour une meilleure lisibilité
    plt.tight_layout()  # Ajuste automatiquement les marges
    
    plt.show()

def plot_vega_surface(ticker, r=0.02):
    data = volatility_surface(ticker, r=r)
    surface = data.pivot_table(values='vega', index='strike', columns='maturity').dropna()

    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(111, projection='3d')
    x, y, z = surface.columns.values, surface.index.values, surface.values

    X, Y = np.meshgrid(x,y)

    ax.set_xlabel("Days to expiration")
    ax.set_ylabel("Strike price")
    ax.set_zlabel("Vega (%)")
    ax.set_title("Vega Surface")

    surf = ax.plot_surface(X, Y, z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

def plot_vega_heatmap(ticker, r=0.02):
    data = volatility_surface(ticker, r=r)
    
    surface = data.pivot_table(values='vega', index='strike', columns='maturity').dropna()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        surface,
        annot=False,  # Si vous voulez afficher les valeurs, mettez True
        cmap='viridis',
        cbar_kws={'label': 'Vega (%)'},  # Légende pour la barre de couleurs
        linewidths=0.5  # Ajoutez des lignes pour délimiter les cases
    )
    
    plt.title("Vega Heatmap", fontsize=16)
    plt.xlabel("Days to Expiration")
    plt.ylabel("Strike Price")
    plt.xticks(rotation=45)  # Rotation des ticks pour une meilleure lisibilité
    plt.tight_layout()  # Ajuste automatiquement les marges
    
    plt.show()

# Get rikfree-rate and implied volatility
def calibration(ticker, K, T):
    S0=yf.download(ticker).iloc[-1]['Close']

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

    return S0, r, IV

def get_price(ticker, K, T, fonction, opt_type='C', M=100000, antithetic=False):
    S0, r, IV = calibration(ticker, K, T)
    price=fonction(S0=S0, K=K, T=T, r=r, vol=IV, opt_type=opt_type, M=M, antithetic=antithetic)
    return price