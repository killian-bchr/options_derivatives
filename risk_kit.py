import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize
import seaborn as sns

def get_ind_size():
    ind = pd.read_csv("C:/Users/Bouchereau Killian/OneDrive/Documents/Certif/MOOC Python/data/ind30_m_size.csv", header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    ind = pd.read_csv("C:/Users/Bouchereau Killian/OneDrive/Documents/Certif/MOOC Python/data/ind30_m_nfirms.csv", header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns():
    ind = pd.read_csv("C:/Users/Bouchereau Killian/OneDrive/Documents/Certif/MOOC Python/data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def annualized_return (ret):
    """Take a DataFrame of asset retuns.
    return the annualized return for each asset
    """
    n_months=ret.shape[0]
    annualized_ret = (ret+1).prod()**(12/n_months)-1
    return annualized_ret

def annualized_volatility (ret):
    """Take a DataFrame of asset retuns.
    return the annualized volatility for each asset
    """
    annualized_vol = ret.std(ddof=0)*np.sqrt(12)
    return annualized_vol

def semideviation(df):
    """Take a DataFrame of asset returns.
    return the semideviation of each asset
    """
    return df[df<0].std(ddof=0)

def sharpe_ratio (ret, risk_free_rate):
    """Take a DataFrame of asset retuns.
    return the sharpe ratio for each asset
    """
    annualized_ret = annualized_return(ret)
    annualized_vol = annualized_volatility(ret)
    excess_return = annualized_ret-risk_free_rate
    return excess_return/annualized_vol
    

def drawdown (ret : pd.Series):
    """Take a time series of asset returns.
    return a DataFrame with columns for the wealth index, 
    the previous peaks and the percentage drawdown
    """
    wealth_index = (1+ret).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth":wealth_index, 
                         "Previous peaks":previous_peaks, 
                         "Drawdown":drawdown}, index=ret.index)

def skewness (df):
    """Alternative to scipy.stats.skewness()
    Compute the skewness of a Series or DataFrame
    """
    excess = df - df.mean()
    excess3 = excess**3
    return excess3.mean()/(df.std(ddof=0)**3)

def kurtosis (df):
    """Alternative to scipy.stats.kurtosis()
    Compute the kurtosis of a Series or DataFrame
    """
    excess = df - df.mean()
    excess4 = excess**4
    return excess4.mean()/(df.std(ddof=0)**4)

def is_normal (r, level=0.01):
    """Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic (r, level = 5):
    """VaR Historic
    """
    if isinstance (r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance (r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected to be Series or DataFrame")

def var_gaussian (r, level=5, modified=False):
    """Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    #compute the z-score assuming it was Gaussian
    z=norm.ppf(level/100)
    if modified:
        #modified the z-score based on observed skewness and kurtosis
        s=skewness(r)
        k=kurtosis(r)
        z = (z + (z**2-1)*(s/6) + (z**3-3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36 )
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """Computes the conditional Var of a Series or DataFrame
    """
    if isinstance (r, pd.Series):
        is_beyond = r<= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected to be Series or DataFrame")
        
def portfolio_return (weight, returns): 
    return weight.T @ returns

def portfolio_vol(weight, cov):
    return (weight.T @ cov @ weight)**0.5

def plot_ef_2(n_points, er, cov, style=".-"):
    """Plots the 2-asset efficient frontier
    """
    weights = [np.array([w,1-w])for w in np.linspace(0,1,n_points)]
    rets=[portfolio_return(w,er)for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({"Returns":rets, "Volatility":vols})
    return ef.plot.line(x="Volatility", y="Returns", style=style)

def target_is_met(target_return, w, er):
    return target_return - portfolio_return(w,er)

def minimize_vol(target_return, er, cov):
    """ Target return --> Weight vector
    """
    n=er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type':'eq',
        'args':(er,),
        'fun': lambda weights, er : target_return - portfolio_return(target_return, weights, er)
    }
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_vol, init_guess, args=(cov,), method='SLSQP', options={'disp' : False},
                       constraints=(return_is_target, weights_sum_to_1), bounds=bounds)
    return results.x

def optimal_weights(n_points, er, cov):
    """list of weihgts to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights=[minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def msr (riskfree_rate, er, cov):
    """Returns the weights of the portfolio that gives you the maximum sharpe ratio 
    given the riskfree rate and expected returns and a covariance matrix
    """
    n=er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """Returns the negative of the sharpe ratio, given weights
        """
        r=portfolio_return(weights, er)
        vol=portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess, args=(riskfree_rate, er, cov,), method='SLSQP',
                       options={'disp' : False},
                       constraints=(weights_sum_to_1), bounds=bounds)
    return results.x

def gmv(cov):
    """Returns the weights of the Global Minimum Vol portfolio
    given the covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)
    
def plot_ef(n_points, er, cov, show_cml=False, style='.-', riskfree_rate=0, show_ew=False, show_gmv=False):
    """Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets=[portfolio_return(w,er)for w in weights]
    vols=[portfolio_vol(w,cov)for w in weights]
    ef=pd.DataFrame({"Returns":rets, "Volatility":vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=True)
    if show_ew:
        n=er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=10, label="EW")
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10, label="GMV")
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', markersize=10, linewidth=2, label="CML")
    ax.legend()
    plt.show()

        
def run_cppi (risky_r,safe_r=None,m=3,start=1000,floor=0.8,riskfree_rate=0.03, drawdown=None):
    dates = risky_r.index
    n_steps = len(dates)
    account_value=start
    floor_value=start*floor
    peak=start
    if isinstance(risky_r,pd.Series):
        risky_r=pd.DataFrame(risky_r, columns=["R"])
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=riskfree_rate/12
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value-floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w,1)
        risky_w = np.maximum(risky_w,0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        ## update the account value for this time step
        account_value = risky_alloc*(1+risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
        ## save the values
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_w
        account_history.iloc[step]=account_value
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result={
        "Wealth":account_history,
        "Risky Wealth":risky_wealth,
        "Risk Budget":cushion_history,
        "Risky Allocation":risky_w_history,
        "m":m,
        "start":start,
        "floor":floor,
        "risky_r":risky_r,
        "safe_r":safe_r
    }
    return backtest_result

def summary_stats(r,riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r=r.aggregate(annualized_return)
    ann_vol=r.aggregate(annualized_volatility)
    ann_sr=r.aggregate(sharpe_ratiorisk_free_rate=riskfree_rate)
    dd=r.aggregate(lambda r:drawdown(r).Drawdown.min())
    skew=r.aggregate(skewness)
    kurt=r.aggregate(kurtosis)
    cf_var5=r.aggregate(var_gaussian, modified=True)
    hist_cvar5=r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return":ann_r,
        "Annualized Vol":ann_vol,
        "Skewness":skew,
        "Kurtosis":kurt,
        "Cornish-Fisher VaR (5%)":cf_var5,
        "Historic CVaR (5%)":hist_cvar5,
        "Sharpe Ratio":ann_sr,
        "Max Drawdown":dd
    })

def gbm0(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    """
    Evolution of a Stock Price using Geometric Brownian Motion Model
    """
    dt=1/steps_per_year
    n_steps=int(n_years*steps_per_year)
    xi=np.random.normal(size=(n_steps, n_scenarios))
    rets=mu*dt+sigma*np.sqrt(dt)*xi
    rets=pd.dataFrame()
    #to prices
    prices = s_0*(1+rets).cumprod()
    return prices
    
def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of a Stock Price using Geometric Brownian Motion Model
    """
    dt=1/steps_per_year
    n_steps=int(n_years*steps_per_year)
    rets_plus_1=np.random.normal(loc=(1+mu*dt),scale=(sigma*np.sqrt(dt)),size=(n_steps+1, n_scenarios))
    rets_plus_1[0]=1
    #to prices
    if prices:
        prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
        return prices
    else:
        return rets_plus_1-1
        
def show_gbm(n_scenarios, mu, sigma):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
    s_0 = 100
    prices=gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax=prices.plot(legend=False, color="indianred", alpha=0.5, linewidth=2, figsize=(12,5))
    ax.axhline(y=s_0, ls=":", color="black")
    ax.set_ylim(top=400)
    #draw a dot at the origin
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)
    
def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100):
    """
    Plot the results of a Monte Carlo simulation of CPPI
    """
    start=100
    sim_rets=gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, steps_per_year=12, prices=False)
    risky_r=pd.DataFrame(sim_rets)
    #run the "back"-test
    btr=run_cppi(risky_r=pd.DataFrame(risky_r), m=m, start=start, floor=floor, riskfree_rate=riskfree_rate)
    wealth=btr["Wealth"]
    
    #calculate terminal wealth stats
    y_max=wealth.values.max()*y_max/100
    terminal_wealth=wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios
    
    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures>0 else 0.0
    
    #Plot ! 
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24,9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9), xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85), xycoords='axes fraction', fontsize=24)
    if floor>0.01:
        hist_ax.axhline(y=start*floor, ls='--', color="red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(.7, .7), xycoords='axes fraction', fontsize=24)

def discount(t,r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    """
    discounts=pd.DataFrame([(r+1)**i for i in t])
    discounts.index=t
    return discounts


def pv (flows,r):
    """Computes the present value of a sequence of cash flows given by the time (as the index) and the amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    returns the present value of the sequence
    """
    dates=flows.index
    discounts=discount(dates,r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """Computes the funding ratio of some assets given liabilities and interest rate
    """
    return pv(assets, r)/pv(liabilities, r)

def show_funding_ratio(assets, liabilities, r):
    fr=funding_ratio(assets, liabilities, r)
    print(f'{fr*100:.2f}')
    
def inst_to_ann(r):
    """Converts short rate to annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """Converts annualized rate to short rate
    """
    return np.log1p(r)

def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None : r_0=b
    r_0 = ann_to_inst(r_0)
    dt=1/steps_per_year
    num_steps = int(n_years*steps_per_year)+1
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates=np.empty_like(shock)
    rates[0]=r_0
    
    ##For price generation
    h=math.sqrt(a**2 + 2*sigma**2)
    prices=np.empty_like(shock)
    
    def price(ttm, r):
        _A=((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B=(2*(math.exp(h*ttm)-1))/(2*h+(h+a)*(math.exp(h*ttm)-1))
        _P=_A*np.exp(-_B*r)
        return _P
    prices[0]=price(n_years,r_0)
                
    for step in range(1, num_steps):
        r_t=rates[step-1]
        d_r_t=a*(b-r_t)*dt+sigma*np.sqrt(r_t)*shock[step]
        rates[step]=abs(r_t+d_r_t)
        ##generate prices at time t as well
        prices[step]=price(n_years -step*dt, rates[step])
    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ###For prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def show_cir(r_0=0.03, a=0.5, b=0.03, sigma=0.05, n_scenarios=5):
    cir(r_0=r_0, a=a, b=b, sigma=sigma, n_scenarios=n_scenarios)[1].plot(legend=False, figsize=(12,5))

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns a series of cash-flows generated by a bond, 
    indexed by a coupon number
    """
    n_coupons=round(maturity*coupons_per_year)
    coupon_amt=principal*coupon_rate/coupons_per_year
    coupon_times=np.arange(1, n_coupons+1)
    cash_flows=pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1]+=principal
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather, it is to illustrate the underlying principle behind bond pricing !
    If discount_rate is a DataFrame, then this is assume to be the rate on each coupon date and the bond value is computed over time. 
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance (discount_rate, pd.DataFrame):
        pricing_dates=discount_rate.index
        prices=pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t]=bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else :
        if maturity <=0 : return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cashflows
    """
    discounted_flows=discount(flows.index, discount_rate)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(pd.DataFrame(flows.index), weights=weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period and that dividends are reinvested in the bond
    """
    coupons=pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max=monthly_prices.index.max()
    pay_date=np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date]=principal*coupon_rate/coupons_per_year
    total_returns=(monthly_prices+coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between two sets of returns 
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces an allocation
    to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
    weights=allocator(r1,r2,**kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that dont match r1")
    r_mix=weights*r1+(1-weights)*r2
    return r_mix

def fixedmix_allocator(r1,r2,w1,**kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Returns the final values of a dollar at the end of the return period
    """
    return (rets+1).prod()

def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Produces summary statistics on the terminal values per invested dollar across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the timestep (we assumed rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth<floor
    reach=terminal_wealth>=cap
    p_breach=breach.mean() if breach.sum()>0 else np.nan
    p_reach=reach.mean() if reach.sum()>0 else np.nan
    e_short=(floor-terminal_wealth[breach]).mean() if breach.sum()>0 else np.nan
    e_surplus=(cap-terminal_wealth[reach]).mean() if reach.sum()>0 else np.nan
    sum_stats=pd.DataFrame.from_dict({
        "mean":terminal_wealth.mean(),
        "std":terminal_wealth.std(),
        "p_breach":p_breach,
        "e_short":e_short, 
        "p_reach":p_reach,
        "e_surplus":e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    """
    n_points=r1.shape[0]
    n_col=r1.shape[1]
    path=pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths=pd.concat([path]*n_col, axis=1)
    paths.index=r1.index
    paths.columns=r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between GHP and PSP with the goal to provide exposure to the upside of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgetint algorithm by investing a multiple of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value=np.repeat(1, n_scenarios)
    floor_value=np.repeat(1, n_scenarios)
    w_history=pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range (n_steps):
        floor_value=floor*zc_prices.iloc[step] ##PV of Floor assuming today's rates and flat YC
        cushion=(account_value-floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) #same as applying min and max
        ghp_w=1-psp_w
        psp_alloc=account_value*psp_w
        ghp_alloc=account_value*ghp_w
        #recompute the new account value at the end of this step
        account_value=psp_alloc*(1+psp_r.iloc[step])+ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step]=psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between GHP and PSP with the goal to provide exposure to the upside of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgetint algorithm by investing a multiple of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value=np.repeat(1, n_scenarios)
    floor_value=np.repeat(1, n_scenarios)
    peak_value=np.repeat(1, n_scenarios)
    w_history=pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range (n_steps):
        floor_value=(1-maxdd)*peak_value ###Floor is based on previous peak
        cushion=(account_value-floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) #same as applying min and max
        ghp_w=1-psp_w
        psp_alloc=account_value*psp_w
        ghp_alloc=account_value*ghp_w
        #recompute the new account value at the end of this step
        account_value=psp_alloc*(1+psp_r.iloc[step])+ghp_alloc*(1+ghp_r.iloc[step])
        peak_value=np.maximum(peak_value, account_value)
        w_history.iloc[step]=psp_w
    return w_history