import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, date, timedelta
from scipy.stats import norm
from sys import exit
from scipy.optimize import least_squares

### Computation of the greeks with Black-Scholes model

def black_scholes(r, S, K, T, sigma, type="c"):
    
    d1=(np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    
    try:
        if type=="c":
            price=S*norm.cdf(d1, 0, 1)-K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type=="p":
            price=K*np.exp(-r*T)*norm.cdf(-d2, 0, 1)-S*norm.cdf(-d1, 0, 1)
        return price
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put")
        
def delta_calc(r, S, K, T, sigma, type="c"):
    d1=(np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    try:
        if type=="c":
            delta_calc=norm.cdf(d1, 0, 1)
        elif type=="p":
            delta_calc=-norm.cdf(-d1, 0, 1)
        return delta_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put")

def gamma_calc(r, S, K, T, sigma, type="c"):
    d1=(np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    try:
        if type=="c":
            gamma_calc=norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))
        elif type=="p":
            gamma_calc=norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))
        return gamma_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put")
        
def vega_calc(r, S, K, T, sigma, type="c"):
    d1=(np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    try:
        if type=="c":
            vega_calc=S*norm.pdf(d1, 0, 1)*np.sqrt(T)
        elif type=="p":
            vega_calc=S*norm.pdf(d1, 0, 1)*np.sqrt(T)
        return vega_calc*0.01   #for 1% change in volatility
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put")
        
def theta_calc(r, S, K, T, sigma, type="c"):
    d1=(np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    try:
        if type=="c":
            theta_calc=-(S*norm.pdf(d1, 0, 1)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type=="p":
            theta_calc=-(S*norm.pdf(d1, 0, 1)*sigma)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return theta_calc/365    #percentage change for 1 day
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put")
        
def rho_calc(r, S, K, T, sigma, type="c"):
    d1=(np.log(S/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    try:
        if type=="c":
            rho_calc=K*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type=="p":
            rho_calc=-K*T*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return rho_calc*0.01   #for 1% change in interest rate
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put")
        
        
### Get the implied volatility
def implied_vol_newton(S, K, T, r, option_price, vol_init=0.3, epsilon=0.00001):
    """Calculate the implied volatility of an European option with the Newton-Raphson method"""
    vol_new=None
    for i in range (500):
        bs_price=black_scholes(r, S, K, T, vol_init)
        vega=vega_calc(r, S, K, T, vol_init)*100   ##we don't want the vega for 1% change in volatility
        if vega !=0:
            vol_new=vol_init-(bs_price-option_price)/vega
        else:
            break
        new_bs_price=black_scholes(r, S, K, T, vol_new)
        
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
            bs_price=black_scholes(r, S, K, T, sigma)
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
    exp_dates=get_options_dates(ticker)
    time_to_maturity=[]
    strike=[]
    implied_volatility=[]
    for date in exp_dates:
        strikes=call(ticker, date)['strike']
        option_prices=call(ticker, date)['lastPrice']
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
        
        
##Get option informations
        
def get_options_dates(ticker_symbol):
    """Take a ticker and returns the list of the existing options expiration dates for this ticker"""
    ticker=yf.Ticker(ticker_symbol)
    return ticker.options
    
def call(ticker_symbol, expiration_date):
    """Take a ticker and an expiration date and returns a dataframe with call option informations related to this ticker for 
        this expiration date"""
    ticker=yf.Ticker(ticker_symbol)
    options_datas=ticker.option_chain(expiration_date)
    return options_datas.calls.drop(columns=['change', 'percentChange', 'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency'])

def put(ticker_symbol, expiration_date):
    """Take a ticker and an expiration date and returns a dataframe with put option informations related to this ticker for 
        this expiration date"""
    ticker=yf.Ticker(ticker_symbol)
    options_datas=ticker.option_chain(expiration_date)
    return options_datas.puts.drop(columns=['change', 'percentChange', 'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency'])

def moneyness_call_option(ticker_symbol, expiration_date):
    """Take a dataframe of call options and returns the dataframe with a column which indicates if these options are in the money, out the money or at the money"""
    calls=call(ticker_symbol, expiration_date)
    strikes=calls['strike']
    ticker=yf.Ticker(ticker_symbol)
    current_data=ticker.history(period='1d')
    current_price=current_data['Close'].iloc[-1]
    def evaluate_strike(strike):
        if strike>current_price:
            return 'OTM'
        elif strike<current_price:
            return 'ITM'
        else:
            return 'ATM'
    calls['moneyness']=calls['strike'].apply(evaluate_strike)
    return calls

def moneyness_put_option(ticker_symbol, expiration_date):
    """Take a dataframe of call options and returns the dataframe with a column which indicates if these options are in the money, out the money or at the money"""
    puts=put(ticker_symbol, expiration_date)
    strikes=puts['strike']
    ticker=yf.Ticker(ticker_symbol)
    current_data=ticker.history(period='1d')
    current_price=current_data['Close'].iloc[-1]
    def evaluate_strike(strike):
        if strike>current_price:
            return 'ITM'
        elif strike<current_price:
            return 'OTM'
        else:
            return 'ATM'
    puts['moneyness']=puts['strike'].apply(evaluate_strike)
    return puts


##Get the price of options using BS and the Greeks associated with the option

def get_prices_per_strike(ticker, maturity_date, risk_free_rate=0.04, type="c", start_date="2020-01-01"):
    df=yf.download(ticker, start=start_date)
    current_price=df['Close'].iloc[-1]
    sigma=np.sqrt(252)*df['Close'].pct_change().rolling(21).std().iloc[-1]
    maturity=(datetime.strptime(maturity_date, '%Y-%m-%d').date()-date.today()).days/365
    prices=[]
    real_prices=[]
    difference=[]
    delta=[]
    gamma=[]
    vega=[]
    theta=[]
    rho=[]
    try:
        if type=="c":
            calls=call(ticker,maturity_date)
            strikes=calls[calls['strike']<200]['strike']
        elif type=="p":
            puts=put(ticker,maturity_date)
            strikes=puts['strike']
        i=0
        for strike in strikes:
            price=black_scholes(risk_free_rate, current_price, strike, maturity, sigma, type=type)
            if type=="c":
                real_prices.append(calls['lastPrice'].iloc[i])
            elif type=="p":
                real_prices.append(puts['lastPrice'].iloc[i])
            prices.append(price)
            difference.append((abs(real_prices[i]-prices[i])/real_prices[i])*100)
            delta.append(delta_calc(risk_free_rate, current_price, strike, maturity, sigma, type=type))
            gamma.append(gamma_calc(risk_free_rate, current_price, strike, maturity, sigma, type=type))
            vega.append(vega_calc(risk_free_rate, current_price, strike, maturity, sigma, type=type))
            theta.append(theta_calc(risk_free_rate, current_price, strike, maturity, sigma, type=type))
            rho.append(rho_calc(risk_free_rate, current_price, strike, maturity, sigma, type=type))
            i+=1
        option_infos=pd.DataFrame({'BS Options prices':prices, 'Real Options prices':real_prices, 'Difference (%)':difference, 
                          'Delta':delta, 'Gamma':gamma, 'Vega':vega, 'Theta':theta, 'Rho':rho}, index=strikes)
        return option_infos
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put")
        
##Computes the BS prices and Greeks for each period

def get_prices_per_period(ticker, maturity_date, strike, risk_free_rate=0.04, type="c", start_date="2022-01-01"):
    df=yf.download(ticker, start=start_date)
    sigma=np.sqrt(252)*df['Close'].pct_change().rolling(21).std().iloc[-1]
    maturity=(datetime.strptime(maturity_date, '%Y-%m-%d').date()-date.today()).days/365
    prices=[]
    delta=[]
    gamma=[]
    vega=[]
    theta=[]
    rho=[]
    for j in df.index:
        price=black_scholes(risk_free_rate, df['Close'].loc[j], strike, maturity, sigma, type=type)
        prices.append(price)
        delta.append(delta_calc(risk_free_rate, df['Close'].loc[j], strike, maturity, sigma, type=type))
        gamma.append(gamma_calc(risk_free_rate, df['Close'].loc[j], strike, maturity, sigma, type=type))
        vega.append(vega_calc(risk_free_rate, df['Close'].loc[j], strike, maturity, sigma, type=type))
        theta.append(theta_calc(risk_free_rate, df['Close'].loc[j], strike, maturity, sigma, type=type))
        rho.append(rho_calc(risk_free_rate, df['Close'].loc[j], strike, maturity, sigma, type=type))       
    option_infos=pd.DataFrame({'Underlying Prices': df['Close'],'BS Options Prices':prices, 'Delta':delta, 'Gamma':gamma, 
                           'Vega':vega, 'Theta':theta, 'Rho':rho,}, index=df.index)
    return option_infos
    
    
#Delta Hedging Option
def delta_hedging(ticker, maturity_date, strike, risk_free_rate=0.04, type="c", start_date="2022-01-01"):
    prices=get_prices_per_period(ticker, maturity_date, strike, risk_free_rate=0.04, type="c", start_date="2022-01-01")
    pnl=prices['BS Options Prices']-prices['BS Options Prices'].iloc[0]
    pnl_delta_position=prices['Delta']*(prices['Underlying Prices']-prices['Underlying Prices'].iloc[0])
    pnl_delta_hedged=(1/2)*prices['Gamma']*((prices['Underlying Prices']-prices['Underlying Prices'].iloc[0])**2)

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,10))

    ax1.plot(prices['Underlying Prices'], pnl, color="blue", label='P&L Option')
    ax1.plot(prices['Underlying Prices'], pnl_delta_position, color="red", label='P&L Delta Position')
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.set_xlabel('Underlying Prices')
    ax1.set_ylabel('P&L')
    ax1.legend(loc='upper left')

    ax2.plot(prices['Underlying Prices'], pnl_delta_hedged, color="green")
    ax2.set_xlabel('Underlying Prices')
    ax2.set_ylabel('P&L')
    ax2.set_title('P&L Delta Hedged Option')

    plt.tight_layout()
    plt.show()
    
    
##Calculate the profit for a covered call

def portfolio_covered_call(ticker, start_date, strike):
    df=yf.download(ticker, start=start_date)
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    expiration_dates=get_options_dates(ticker)
    initial_price=df.loc[start_date, 'Close']
    portfolios = pd.DataFrame(index=expiration_dates, columns=['Option price','Day 1'])
    for exp_date in expiration_dates:
        buy_call=call(ticker,exp_date)
        option_price=buy_call[buy_call['strike']==strike]['lastPrice']
        portfolios.loc[exp_date, 'Option price']=option_price.values[0]
        portfolios.loc[exp_date, 'Day 1']=-df.loc[start_date,'Close']+option_price.values[0]
    
    #following_date=start_date+timedelta(days=1)
    #df_=df[following_date:]
    #i=2
    #for price in df_['Close']:
    #    column_name = f"Day{i}"
    #    if price>strike:
    #        portfolios[column_name]=-strike+portfolios["Option price"]
    #    else:
    #        portfolios[column_name]=portfolios["Option price"]-price
    #    i+=1
    return portfolios

def return_covered_call(ticker, start_date, strike):
    df=yf.download(ticker, start=start_date)
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    initial_price=df.loc[start_date, 'Close']
    date=datetime.strptime(start_date, "%Y-%m-%d")+timedelta(days=1)
    portfolios=portfolio_covered_call(ticker, start_date, strike)
    expiration_dates=get_options_dates(ticker)
    profits=pd.DataFrame(index=expiration_dates, columns=['Profit'])
    for i in portfolios.index:
        while date != i:
            if date>df.index[-1]:
                for k in portfolios.index:
                    if date<datetime.strptime(k, "%Y-%m-%d"):
                        profits.loc[k,'Profit']=np.nan
                break
            j=2
            column_name = f"Day{j}"
            if date in df.index:
                if df.loc[date, 'Close']>strike:
                    portfolios[column_name]=-strike+portfolios["Option price"]
                else:
                    portfolios[column_name]=portfolios["Option price"]-df.loc[date, 'Close']
            else:
                last_column=portfolios.iloc[:,-1]
                portfolios[column_name]=last_column  
            j+=1
            date=date+timedelta(days=1)

        if datetime.strptime(i, "%Y-%m-%d")>df.index[-1]:
            if date<datetime.strptime(i, "%Y-%m-%d"):
                profits.loc[i,'Profit']=np.nan
            break
        ##if df.loc[(date-timedelta(days=1)).strftime("%Y-%m-%d"), 'Close']>strike:    on revient au jour d'avant
        else:
            if df.loc[i, 'Close']>strike:
                profit=portfolios.loc[i][-1]+strike
            else:
                #profit=portfolios.loc[i][-1]+df.loc[(date-timedelta(days=1)).strftime("%Y-%m-%d"), 'Close']
                profit=portfolios.loc[i][-1]+df.loc[i, 'Close']
        duration=(date-datetime.strptime(start_date, "%Y-%m-%d")).days
        #portfolios.drop(index=i, inplace=True)
        return_=(profit/initial_price)*(365/duration)
        profits.loc[i, 'Profit']=return_
    return profits
        
##Show the payoffs

def buy_call_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    calls=call(ticker_symbol, expiration_date)
    if strike in calls['strike'].values:
        option=calls[calls['strike']==strike]
        premium=option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(current_price-1.5*current_price, current_price+1.5*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]>strike:
                payoff.append(prices[i]-strike-premium)
            else:
                payoff.append(-premium)
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")

def show_buy_call_payoff(ticker_symbol, expiration_date):
    calls=call(ticker_symbol, expiration_date)
    ##define the current price of the underlying asset
    ticker=yf.Ticker(ticker_symbol)
    current_data=ticker.history(period='1d')
    current_price=current_data['Close'].iloc[-1]
    prices=np.linspace(current_price-1.5*current_price, current_price+1.5*current_price, 1000)
    strikes=calls['strike']
    payoffs=[]
    for i in range (len(calls)):
        strike=calls['strike'].iloc[i]
        premium=calls['lastPrice'].iloc[i]
        payoff=[]
        for i in range (len(prices)):
            if prices[i]>strike:
                payoff.append(prices[i]-strike-premium)
            else:
                payoff.append(-premium)
        payoffs.append(payoff)
        
    plt.figure(figsize=(12, 8))
    j=0
    for payoff in payoffs:
        plt.plot(prices, payoff, linewidth=2, label=f'Strike : {strikes.iloc[j]}')
        j+=1
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Underlying Asset Prices')
    plt.ylabel('Payoff')
    plt.legend()
    plt.show()

def buy_put_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    puts=put(ticker_symbol, expiration_date)
    if strike in puts['strike'].values:
        option=puts[puts['strike']==strike]
        premium=option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(current_price-1.5*current_price, current_price+1.5*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]<strike:
                payoff.append(strike-prices[i]-premium)
            else:
                payoff.append(-premium)
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")
        
def sell_call_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    calls=call(ticker_symbol, expiration_date)
    if strike in calls['strike'].values:
        option=calls[calls['strike']==strike]
        premium=option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(current_price-1.5*current_price, current_price+1.5*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]<strike:
                payoff.append(premium)
            else:
                payoff.append(premium-(prices[i]-strike))
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")
        
def sell_put_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    puts=put(ticker_symbol, expiration_date)
    if strike in puts['strike'].values:
        option=puts[puts['strike']==strike]
        premium=option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(current_price-1.5*current_price, current_price+1.5*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]>strike:
                payoff.append(premium)
            else:
                payoff.append(prices[i]-strike+premium)
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")
        
def buy_straddle_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    calls=call(ticker_symbol, expiration_date)
    puts=put(ticker_symbol, expiration_date)
    if strike in puts['strike'].values and strike in calls['strike'].values:
        call_option=calls[calls['strike']==strike]
        put_option=puts[puts['strike']==strike]
        call_premium=call_option['lastPrice'].values
        put_premium=put_option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(strike-2*current_price, strike+2*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]>strike:
                payoff.append(prices[i]-strike-put_premium-call_premium)
            else:
                payoff.append(strike-prices[i]-put_premium-call_premium)
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")
        
def sell_straddle_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    calls=call(ticker_symbol, expiration_date)
    puts=put(ticker_symbol, expiration_date)
    if strike in puts['strike'].values and strike in calls['strike'].values:
        call_option=calls[calls['strike']==strike]
        put_option=puts[puts['strike']==strike]
        call_premium=call_option['lastPrice'].values
        put_premium=put_option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(strike-2*current_price, strike+2*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]>strike:
                payoff.append(put_premium+call_premium-(prices[i]-strike))
            else:
                payoff.append(put_premium+call_premium-(strike-prices[i]))
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")
        
        
def long_synthetic_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    calls=call(ticker_symbol, expiration_date)
    puts=put(ticker_symbol, expiration_date)
    if strike in puts['strike'].values and strike in calls['strike'].values:
        call_option=calls[calls['strike']==strike]
        put_option=puts[puts['strike']==strike]
        call_premium=call_option['lastPrice'].values
        put_premium=put_option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(strike-2*current_price, strike+2*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]>strike:
                payoff.append(prices[i]-strike-call_premium+put_premium)
            else:
                payoff.append(put_premium-call_premium-(strike-prices[i]))
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")
                              
def short_synthetic_payoff(ticker_symbol, strike, expiration_date):
    ##get the informations of option with the given strike
    calls=call(ticker_symbol, expiration_date)
    puts=put(ticker_symbol, expiration_date)
    if strike in puts['strike'].values and strike in calls['strike'].values:
        call_option=calls[calls['strike']==strike]
        put_option=puts[puts['strike']==strike]
        call_premium=call_option['lastPrice'].values
        put_premium=put_option['lastPrice'].values
        ##define the current price of the underlying asset
        ticker=yf.Ticker(ticker_symbol)
        current_data=ticker.history(period='1d')
        current_price=current_data['Close'].iloc[-1]
        ##compute the payoff of the option
        prices=np.linspace(strike-2*current_price, strike+2*current_price, 1000)
        payoff=[]
        for i in range (len(prices)):
            if prices[i]>strike:
                payoff.append(strike-prices[i]-put_premium+call_premium)
            else:
                payoff.append(call_premium-put_premium-(prices[i]-strike))
        plt.figure(figsize=(12, 6))
        plt.plot(prices, payoff, color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--')
        #plt.axvline(x=current_price, color='blue', linestyle='--', alpha=0.5)
        plt.xlabel('Underlying Asset Prices')
        plt.ylabel('Payoff')
        plt.show()
    else:
        print("This strike price doesn't exist")
        
def display_call_strikes(ticker_symbol, expiration_date):
    """Take a ticker symbol and expiration date and returns the moneyness of different strikes prices for call options"""
    calls=moneyness_call_option(ticker_symbol, expiration_date)
    ATM=[]
    ITM=[]
    OTM=[]
    for i in range (len(calls)):
        if calls['moneyness'].iloc[i] == 'ITM':
            ITM.append(calls['strike'].iloc[i])
        elif calls['moneyness'].iloc[i] == 'ATM':
            ATM.append(calls['strike'].iloc[i])
        elif calls['moneyness'].iloc[i] == 'OTM':
            OTM.append(calls['strike'].iloc[i])
        else:
            print("The option isn't evaluate")
            
    max_length = max(len(ITM), len(ATM), len(OTM))
    ITM.extend([np.nan] * (max_length - len(ITM)))
    ATM.extend([np.nan] * (max_length - len(ATM)))
    OTM.extend([np.nan] * (max_length - len(OTM)))
    return pd.DataFrame({'ITM' : ITM, 'ATM' : ATM, 'OTM' : OTM})

def bull_call_spread_payoff(ticker_symbol, expiration_date, buy_strike, sell_strike):
    calls=call(ticker_symbol, expiration_date)
    buy_call_option = calls[calls['strike']==buy_strike]
    sell_call_option = calls[calls['strike']==sell_strike]
    buy_premium = buy_call_option['lastPrice'].values
    sell_premium = sell_call_option['lastPrice'].values
    ##define the current price of the underlying asset
    ticker=yf.Ticker(ticker_symbol)
    current_data=ticker.history(period='1d')
    current_price=current_data['Close'].iloc[-1]
    ##compute the payoff of the option
    prices=np.linspace(buy_strike-current_price, sell_strike+current_price, 1000)
    payoff=[]
    for i in range (len(prices)):
        if prices[i]>sell_strike:
            payoff.append(sell_premium-buy_premium-buy_strike+sell_strike)
        elif prices[i]>buy_strike and prices[i]<sell_strike:
            payoff.append(prices[i]-buy_strike-buy_premium+sell_premium)
        else:
            payoff.append(sell_premium-buy_premium)
    
            
    plt.figure(figsize=(12, 6))
    plt.plot(prices, payoff, color='red', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Underlying Asset Prices')
    plt.ylabel('Payoff')
    plt.show()

def call_backspread_payoff(ticker_symbol, expiration_date, sell_strike, buy_strike):
    calls=call(ticker_symbol, expiration_date)
    buy_call_option = calls[calls['strike']==buy_strike]
    sell_call_option = calls[calls['strike']==sell_strike]
    buy_premium = buy_call_option['lastPrice'].values
    sell_premium = sell_call_option['lastPrice'].values
    ##define the current price of the underlying asset
    ticker=yf.Ticker(ticker_symbol)
    current_data=ticker.history(period='1d')
    current_price=current_data['Close'].iloc[-1]
    ##compute the payoff of the option
    prices=np.linspace(buy_strike-current_price, sell_strike+current_price, 1000)
    payoff=[]
    for i in range (len(prices)):
        if prices[i]>buy_strike:
            payoff.append(prices[i]+sell_premium-2*buy_premium+sell_strike-2*buy_strike)
        elif prices[i]>sell_strike and prices[i]<buy_strike:
            payoff.append(sell_premium-2*buy_premium+sell_strike-prices[i])
        else:
            payoff.append(sell_premium-2*buy_premium)
    
            
    plt.figure(figsize=(12, 6))
    plt.plot(prices, payoff, color='red', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Underlying Asset Prices')
    plt.ylabel('Payoff')
    plt.show()

def long_call_butterfly_payoff(ticker_symbol, expiration_date, ITM_strike, ATM_strike, OTM_strike):
    calls=call(ticker_symbol, expiration_date)
    ITM_option = calls[calls['strike']==ITM_strike]
    ATM_option = calls[calls['strike']==ATM_strike]
    OTM_option = calls[calls['strike']==OTM_strike]
    ITM_premium = ITM_option['lastPrice'].values
    ATM_premium = ATM_option['lastPrice'].values
    OTM_premium = OTM_option['lastPrice'].values
    ##define the current price of the underlying asset
    ticker=yf.Ticker(ticker_symbol)
    current_data=ticker.history(period='1d')
    current_price=current_data['Close'].iloc[-1]
    ##compute the payoff of the option
    prices=np.linspace(ITM_strike-current_price, OTM_strike+current_price, 1000)
    payoff=[]
    for i in range (len(prices)):
        if prices[i]>OTM_strike:
            payoff.append(2*ATM_premium-OTM_premium-ITM_premium-ITM_strike-OTM_strike+2*ATM_strike)
        elif prices[i]>ATM_strike and prices[i]<OTM_strike:
            payoff.append(2*ATM_premium-OTM_premium-ITM_premium-ITM_strike+2*ATM_strike-prices[i])
        elif prices[i]>ITM_strike and prices[i]<ATM_strike:
            payoff.append(2*ATM_premium-OTM_premium-ITM_premium+prices[i]-ITM_strike)
        else:
            payoff.append(2*ATM_premium-OTM_premium-ITM_premium)
    
            
    plt.figure(figsize=(12, 6))
    plt.plot(prices, payoff, color='red', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Underlying Asset Prices')
    plt.ylabel('Payoff')
    plt.show()
    