import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import datetime, timedelta, timezone
from matplotlib.gridspec import GridSpec
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import warnings

# Configuration constants
MAX_RETURN_THRESHOLD = 2.0
MIN_VOLATILITY = 0.0001
MAX_CASH_ALLOCATION = 0.20
SYNTHETIC_VOL = 0.20  # 20% annual volatility for synthetic data

def get_risk_free_rate():
    return 0.0425  # Update this rate as needed

def get_valid_data(tickers, start, end, etf_params):
    valid_tickers = []
    data = pd.DataFrame()
    div_yields = {}
    
    ticker_iterator = tqdm(tickers, desc='Downloading data', unit='ticker', 
                          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    for ticker in ticker_iterator:
        ticker_iterator.set_postfix_str(ticker)
        try:
            stock = yf.Ticker(ticker)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = stock.history(start=start, end=end, auto_adjust=True)
            
            # Enhanced data validation
            expected_days = (end - start).days
            if hist.empty or len(hist) < max(50, 0.3 * expected_days):
                ticker_iterator.write(f"Skipping {ticker} - insufficient data ({len(hist)} days)")
                continue
                
            hist.index = hist.index.tz_localize(None)
            data[ticker] = hist['Close']
            
            div_yield = 0.0
            current_price = hist['Close'].iloc[-1]
            
            if ticker not in etf_params and current_price > 0:
                try:
                    # Improved dividend calculation
                    divs = stock.dividends
                    if not divs.empty:
                        cutoff_date = hist.index[-1] - pd.DateOffset(years=1)
                        last_year_div = divs[divs.index >= cutoff_date].sum()
                        div_yield = last_year_div / current_price
                    else:
                        info = stock.info
                        div_yield = info.get('dividendYield', 0.0)
                        
                except Exception as e:
                    ticker_iterator.write(f"Dividend warning for {ticker}: {str(e)}")
                    div_yield = 0.0
                
                div_yield = min(div_yield, 0.30)
                
            div_yields[ticker] = div_yield
            valid_tickers.append(ticker)
            
        except Exception as e:
            ticker_iterator.write(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not data.empty:
        data['CASH'] = 1.0
    else:
        print("Warning: No valid assets found")
    
    return data, valid_tickers, div_yields

def run_monte_carlo(args):
    expected_returns, cov_matrix, valid_tickers, n_assets, chunk_size, constraints, risk_free_rate = args
    min_weights = np.array([constraints.get(ticker, 0) for ticker in valid_tickers + ['CASH']])
    total_min = min_weights.sum()
    
    dirichlet_weights = np.random.dirichlet(np.ones(n_assets), size=chunk_size)
    remaining = 1 - total_min
    weights = min_weights + dirichlet_weights * remaining
    
    rets = np.dot(weights, expected_returns)
    vols = np.sqrt(np.einsum('...i,ij,...j->...', weights, cov_matrix, weights))
    
    rets = np.clip(rets, -1, MAX_RETURN_THRESHOLD)
    vols = np.clip(vols, MIN_VOLATILITY, None)
    
    sharpes = (rets - risk_free_rate) / np.where(vols > 1e-6, vols, np.nan)
    return np.vstack([rets, vols, sharpes])

def parallel_monte_carlo(expected_returns, cov_matrix, valid_tickers, n_assets, 
                        n_simulations, constraints, risk_free_rate, plot_sample_size):
    num_cores = mp.cpu_count()
    chunk_size = min(n_simulations // (num_cores * 10), 10000)
    chunks = [chunk_size] * (n_simulations // chunk_size)
    remainder = n_simulations % chunk_size
    if remainder > 0:
        chunks.append(remainder)
    
    args = [(expected_returns, cov_matrix, valid_tickers, n_assets, c, 
            constraints, risk_free_rate) for c in chunks]
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(run_monte_carlo, args), 
                      total=len(chunks),
                      desc='Simulating portfolios',
                      unit='chunk'))
    
    full_results = np.hstack(results)
    
    if full_results.shape[1] > plot_sample_size:
        rng = np.random.default_rng()
        indices = rng.choice(full_results.shape[1], plot_sample_size, replace=False)
        return full_results[:, indices]
    return full_results

def optimize_portfolio(expected_returns, cov_matrix, valid_tickers, n_assets, constraints, risk_free_rate):
    def negative_sharpe(weights):
        ret = np.dot(weights, expected_returns)
        vol = np.sqrt(weights @ cov_matrix @ weights.T)
        return -(ret - risk_free_rate)/vol if vol > 1e-6 else np.inf
    
    bounds = [
        (constraints.get(ticker, 0), 
        MAX_CASH_ALLOCATION if ticker == 'CASH' else 0.75)
        for ticker in valid_tickers + ['CASH']
    ]
    
    constraints_list = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: 0.75 - np.max(x)}
    ]
    
    initial_guess = np.array([0.0 if ticker == 'CASH' else 0.05 for ticker in valid_tickers + ['CASH']])
    initial_guess /= initial_guess.sum()
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize._slsqp_py')
        result = minimize(negative_sharpe, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints_list,
                         options={'maxiter': 1000, 'ftol': 1e-9})
    
    return result.x if result.success else None

def create_allocation_charts(weights, valid_tickers, iteration=None):
    threshold = 0.03
    labels = valid_tickers + ['CASH']
    final_weights = weights
    
    small_weights = [(label, w) for label, w in zip(labels, final_weights) if w < threshold]
    main_weights = [(label, w) for label, w in zip(labels, final_weights) if w >= threshold]
    
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    if small_weights:
        gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.4, figure=fig)
        ax_pie = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
    else:
        ax_pie = fig.add_subplot(111)
        ax_table = None
    
    pie_labels = [f"{label}\n{w*100:.1f}%" for label, w in main_weights]
    pie_weights = [w for _, w in main_weights]
    if small_weights:
        pie_labels.append(f"Others\n{sum(w for _, w in small_weights)*100:.1f}%")
        pie_weights.append(sum(w for _, w in small_weights))
    
    explode = [0.1 if w == max(pie_weights) else 0 for w in pie_weights]
    ax_pie.pie(pie_weights, labels=pie_labels, autopct='%1.1f%%', explode=explode,
             wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, startangle=90)
    ax_pie.set_title(f'Portfolio Allocation{"" if iteration is None else f" (Iteration {iteration})"}')

    if small_weights and ax_table:
        ax_table.axis('off')
        table_data = [(label, f"{w*100:.2f}%") for label, w in small_weights]
        table = ax_table.table(cellText=table_data,
                            colLabels=['Asset', 'Weight'],
                            loc='center',
                            cellLoc='center',
                            colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax_table.set_title('Breakdown of "Others" Category')
    plt.show()

def get_ipo_date(ticker):
    """Improved IPO date detection with multiple fallbacks"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try different methods to get IPO date
        ipo_date = stock.info.get('firstTradeDateEpochUtc')
        if ipo_date:
            dt = datetime.fromtimestamp(ipo_date).replace(tzinfo=None)
            if dt.year > 1970:  # Sanity check
                return dt
            
        # Fallback to earliest historical data
        hist = stock.history(period="max")
        if not hist.empty:
            return hist.index[0].to_pydatetime().replace(tzinfo=None)
            
        return datetime(1900, 1, 1)
    except Exception as e:
        print(f"Error fetching IPO date for {ticker}: {e}")
        return datetime(1900, 1, 1)

def backtest_portfolio(weights, assets, start_date, end_date, boe_rate, etf_params, include_dividends, div_yields):
    """Enhanced backtest with proper IPO date handling"""
    all_assets = assets + ['CASH']
    weights = np.array(weights)
    synthetic_assets = []
    ipo_dates = {asset: get_ipo_date(asset) for asset in assets if asset != 'CASH'}
    ipo_dates['CASH'] = datetime(1900, 1, 1)
    
    # Convert all dates to naive datetime objects
    start_date = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
    end_date = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
    
    # Pre-check asset availability
    invalid_assets = []
    for asset in assets:
        if asset == 'CASH':
            continue
        ipo_date = ipo_dates[asset]
        available_date = ipo_date + timedelta(days=730)
        if available_date > end_date:
            invalid_assets.append(f"{asset} (available from {available_date.date()})")
    
    if invalid_assets:
        print(f"\nCannot backtest - these assets are invalid for the period:")
        print("\n".join(invalid_assets))
        return None, None
    
    data = pd.DataFrame()
    for asset in assets:
        if asset == 'CASH':
            continue
            
        if asset in etf_params:
            # Handle ETF synthetic data
            n_days = (end_date - start_date).days + 1
            daily_vol = etf_params[asset]['vol'] / np.sqrt(252)
            price_returns = np.random.normal(0, daily_vol, n_days)
            synthetic_assets.append(asset)
            
            if include_dividends:
                div_contribution = etf_params[asset]['div_yield'] / 252
                synthetic_returns = price_returns + div_contribution
            else:
                synthetic_returns = price_returns
                
            synthetic_prices = np.cumprod(1 + synthetic_returns) * 100
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            data[asset] = pd.Series(synthetic_prices, index=dates)
            continue

        try:
            ipo_date = ipo_dates[asset]
            available_date = ipo_date + timedelta(days=730)
            
            # Get historical data starting from available date
            stock_data = yf.Ticker(asset).history(
                start=available_date - timedelta(days=365),
                end=end_date,
                auto_adjust=True
            )['Close']
            
            # Filter to backtest period
            stock_data = stock_data.loc[start_date:end_date]
            
            if len(stock_data) < 0.8 * (end_date - start_date).days:
                raise ValueError(f"Insufficient data ({len(stock_data)} days)")
                
            stock_data = stock_data[~stock_data.index.duplicated()]
            data[asset] = stock_data
            
        except Exception as e:
            print(f"Using synthetic data for {asset}: {str(e)}")
            n_days = (end_date - start_date).days + 1
            synthetic_returns = np.random.normal(0, SYNTHETIC_VOL/np.sqrt(252), n_days)
            
            if include_dividends and asset in div_yields:
                synthetic_returns += div_yields[asset]/252
                
            synthetic_prices = np.cumprod(1 + synthetic_returns) * 100
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            data[asset] = pd.Series(synthetic_prices, index=dates)
            synthetic_assets.append(asset)
    
    data['CASH'] = 1.0
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = data.reindex(all_dates).ffill().bfill()
    
    returns = pd.DataFrame(index=data.index)
    for asset in all_assets:
        if asset == 'CASH':
            returns['CASH'] = boe_rate/252
        else:
            returns[asset] = data[asset].pct_change(fill_method=None)
            if returns[asset].std() < 1e-6:
                returns[asset] += np.random.normal(0, 1e-5, len(returns))
    
    adjusted_weights = pd.DataFrame(index=returns.index, columns=all_assets)
    
    # Improved weight redistribution logic
    for date in returns.index:
        valid_assets = ['CASH']
        date_naive = date.to_pydatetime().replace(tzinfo=None)
        
        for asset in assets:
            if asset in synthetic_assets:
                continue
                
            ipo_date = ipo_dates[asset]
            available_date = ipo_date + timedelta(days=730)
            
            if (date_naive >= available_date) and pd.notna(returns.loc[date, asset]):
                valid_assets.append(asset)
        
        if len(valid_assets) > 1:
            valid_indices = [all_assets.index(a) for a in valid_assets]
            valid_weights = weights[valid_indices]
            
            # Normalize weights to sum to 1
            total_weight = valid_weights.sum()
            if total_weight > 1e-6:
                valid_weights /= total_weight
            adjusted_weights.loc[date, valid_assets] = valid_weights
        else:
            adjusted_weights.loc[date, 'CASH'] = 1.0
    
    # Clean up weights
    adjusted_weights = adjusted_weights.ffill().infer_objects(copy=False).fillna(0.0)
    adjusted_weights = adjusted_weights.div(adjusted_weights.sum(axis=1), axis=0)
    
    portfolio_returns = (adjusted_weights * returns).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    if synthetic_assets:
        print(f"\nUsed synthetic data for: {', '.join(set(synthetic_assets))}")
    
    return cumulative_returns, portfolio_returns

def plot_backtest_results(cumulative_returns, portfolio_returns, boe_rate):
    plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)
    
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(cumulative_returns * 100, label='Portfolio')
    ax1.set_title('Cumulative Returns (%)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(gs[1, :])
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak)/peak
    ax2.fill_between(drawdown.index, drawdown*100, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdowns (%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(gs[2, 0])
    ax3.axis('off')
    
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (annual_return - boe_rate)/annual_vol
    
    table_data = [
        ["Annual Return", f"{annual_return*100:.1f}%"],
        ["Volatility", f"{annual_vol*100:.1f}%"],
        ["Sharpe Ratio", f"{sharpe:.2f}"],
        ["Max Drawdown", f"{drawdown.min()*100:.1f}%"]
    ]
    
    table = ax3.table(cellText=table_data,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    ax4 = plt.subplot(gs[2, 1])
    monthly_vol = portfolio_returns.resample('ME').std() * np.sqrt(252)
    monthly_vol.tail(24).plot(kind='bar', ax=ax4, color='purple', alpha=0.7)
    ax4.set_title('Recent Volatility (24 Months)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylabel('Annualized Volatility')
    
    plt.show()

def main():
    tickers = [t.strip().upper() for t in input("Enter stock tickers (comma-separated): ").split(',')]
    
    etf_params = {}
    print("\nIdentify ETFs (enter tickers one by one, leave blank when done)")
    while True:
        etf = input("Enter ETF ticker (or press Enter to finish): ").strip().upper()
        if not etf:
            break
        if etf not in tickers:
            print(f"{etf} not in ticker list")
            continue
            
        while True:
            try:
                vol = float(input(f"Enter annual volatility for {etf} (0.05-0.50): "))
                if not 0.05 <= vol <= 0.50:
                    raise ValueError
                break
            except:
                print("Volatility must be between 0.05 and 0.50")
                
        while True:
            try:
                div_yield = float(input(f"Enter annual dividend yield for {etf} (0.00-0.15): "))
                if not 0.00 <= div_yield <= 0.15:
                    raise ValueError
                break
            except:
                print("Dividend yield must be between 0.00 and 0.15")
                
        etf_params[etf] = {'vol': vol, 'div_yield': div_yield}
    
    include_dividends = input("\nInclude dividends in returns? (y/n): ").lower() == 'y'
    boe_rate = get_risk_free_rate()
    print(f"\nUsing current UK 3-month Gilt yield as risk-free rate: {boe_rate*100:.2f}%")
    
    n_simulations = int(input("Enter number of portfolios to simulate (e.g., 1000000): "))
    plot_sample_size = int(input("Enter number of points to plot (e.g., 10000): "))
    
    end_date = datetime.now(timezone.utc).replace(tzinfo=None)
    start_date = end_date - timedelta(days=5*365)
    print(f"\nUsing historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    print("Downloading historical data...")
    data, valid_tickers, div_yields = get_valid_data(tickers, start=start_date, end=end_date, etf_params=etf_params)
    
    if len(valid_tickers) < 1:
        print("Insufficient valid tickers to proceed")
        return
    
    returns = data.pct_change(fill_method=None).dropna()
    
    for etf in etf_params:
        if etf not in returns.columns:
            n_days = len(returns)
            daily_vol = etf_params[etf]['vol'] / np.sqrt(252)
            price_returns = np.random.normal(0, daily_vol, n_days)
            
            if include_dividends:
                div_contribution = etf_params[etf]['div_yield'] / 252
                returns[etf] = price_returns + div_contribution
            else:
                returns[etf] = price_returns
    
    cash_return = pd.Series(boe_rate/252, index=returns.index)
    returns['CASH'] = cash_return
    
    price_returns = returns.mean() * 252
    dividend_returns = []
    
    for asset in returns.columns:
        if asset == 'CASH':
            dividend_returns.append(0.0)
        elif asset in etf_params:
            dividend_returns.append(0.0)
        else:
            try:
                dividend_returns.append(div_yields[asset] if include_dividends else 0.0)
            except KeyError:
                dividend_returns.append(0.0)
    
    expected_returns = np.clip(price_returns.values + np.array(dividend_returns), -0.5, 1.0)
    
    print("\n=== ASSET STATISTICS ===")
    stats_df = pd.DataFrame({
        'Asset': returns.columns,
        'Price Return': price_returns.values,
        'Dividend Yield': dividend_returns,
        'Total Return': expected_returns
    }).sort_values('Total Return', ascending=False)
    print(stats_df.to_string(index=False))
    
    cov_estimator = LedoitWolf(assume_centered=True)
    cov_matrix = cov_estimator.fit(returns).covariance_ * 252
    cov_matrix = pd.DataFrame(cov_matrix, 
                            columns=returns.columns, 
                            index=returns.columns)
    
    for etf in etf_params:
        if etf in cov_matrix.columns:
            etf_std = etf_params[etf]['vol']
            original_std = np.sqrt(cov_matrix.loc[etf, etf])
            if original_std > 0:
                scale_factor = etf_std / original_std
                cov_matrix.loc[etf] *= scale_factor
                cov_matrix.loc[:, etf] *= scale_factor
                cov_matrix.loc[etf, etf] = etf_std**2
            else:
                cov_matrix.loc[etf, etf] = etf_std**2
    
    n_assets = len(valid_tickers) + 1
    current_constraints = {ticker: 0.005 for ticker in valid_tickers}
    iteration = 0
    all_results = []
    final_weights = None
    
    while True:
        iteration += 1
        print(f"\n=== OPTIMIZATION ITERATION {iteration} ===")
        
        print(f"Running {n_simulations} Monte Carlo simulations...")
        mc_results = parallel_monte_carlo(expected_returns, cov_matrix.values, valid_tickers,
                                        n_assets, n_simulations, current_constraints, boe_rate, plot_sample_size)
        
        optimal_weights = optimize_portfolio(expected_returns, cov_matrix.values, valid_tickers,
                                            n_assets, current_constraints, boe_rate)
        
        if optimal_weights is None:
            break
            
        final_weights = optimal_weights
        at_minimum = []
        for ticker, weight in zip(valid_tickers + ['CASH'], optimal_weights):
            min_weight = current_constraints.get(ticker, 0)
            if abs(weight - min_weight) < 0.001:
                at_minimum.append(ticker)
        
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(mc_results[1], mc_results[0], c=mc_results[2], cmap='viridis',
                   marker='o', alpha=0.1, s=10, label='Possible Portfolios')
        ax1.scatter(np.sqrt(optimal_weights @ cov_matrix.values @ optimal_weights.T),
                   np.dot(optimal_weights, expected_returns),
                   c='red', s=200, marker='*', edgecolor='black', label='Optimal Portfolio')
        plt.colorbar(ax1.collections[0], ax=ax1, label='Sharpe Ratio')
        ax1.set_xlabel('Volatility (Risk)')
        ax1.set_ylabel('Expected Return')
        ax1.set_title(f'Portfolio Optimization (Iteration {iteration})')
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=min(0, boe_rate - 0.05))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()
        plt.show()
        
        create_allocation_charts(optimal_weights, valid_tickers, iteration)
        
        print(f"\nITERATION {iteration} OPTIMAL WEIGHTS:")
        for ticker, weight in zip(valid_tickers + ['CASH'], optimal_weights):
            print(f"{ticker}: {weight*100:.2f}%")
        
        adjust = input("\nAdjust constraints for assets at minimum? (y/n): ").lower()
        if adjust != 'y':
            break
            
        print("\nAssets at current minimum allocation:")
        for ticker in at_minimum:
            if ticker in current_constraints:
                print(f"- {ticker} ({current_constraints[ticker]*100:.1f}%)")
                
        for ticker in at_minimum:
            if ticker == 'CASH':
                continue
            response = input(f"Adjust {ticker} constraint? (y/n): ").lower()
            if response == 'y':
                while True:
                    try:
                        new_min = float(input(f"New minimum % for {ticker} (0-100): ")) / 100
                        if 0 <= new_min <= 1:
                            current_constraints[ticker] = max(new_min, current_constraints.get(ticker, 0))
                            break
                        else:
                            print("Please enter a value between 0 and 100")
                    except ValueError:
                        print("Invalid input. Please enter a numeric value")
                
        all_results.append({
            'weights': optimal_weights,
            'constraints': current_constraints.copy(),
            'return': np.dot(optimal_weights, expected_returns),
            'risk': np.sqrt(optimal_weights @ cov_matrix.values @ optimal_weights.T)
        })


    if final_weights is not None:
        print("\n=== FINAL OPTIMAL PORTFOLIO ===")
        
        final_mc_results = parallel_monte_carlo(
            expected_returns, cov_matrix.values, valid_tickers,
            n_assets, 100000, current_constraints, boe_rate, 50000
        )
        
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(final_mc_results[1], final_mc_results[0], c=final_mc_results[2], cmap='viridis',
                   marker='o', alpha=0.1, s=10, label='Possible Portfolios')
        ax1.scatter(np.sqrt(final_weights @ cov_matrix.values @ final_weights.T),
                   np.dot(final_weights, expected_returns),
                   c='red', s=200, marker='*', edgecolor='black', label='Optimal Portfolio')
        plt.colorbar(ax1.collections[0], ax=ax1, label='Sharpe Ratio')
        ax1.set_xlabel('Volatility (Risk)')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Final Optimal Portfolio')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()
        plt.show()
        
        create_allocation_charts(final_weights, valid_tickers)
        

        backtest = input("\nPerform backtest on final portfolio? (y/n): ").lower()
        while backtest == 'y':
            print("\nEnter backtest date range (YYYY-MM-DD format)")
            while True:
                try:
                    start_str = input("Start date (minimum 1990-01-01): ").strip()
                    if start_str:
                        start_date = datetime.strptime(start_str, "%Y-%m-%d")
                        start_date = max(start_date, datetime(1990, 1, 1))
                    else:
                        start_date = datetime(1990, 1, 1)
                    
                    end_str = input("End date (default: yesterday): ").strip()
                    if end_str:
                        end_date = datetime.strptime(end_str, "%Y-%m-%d")
                    else:
                        end_date = datetime.now() - timedelta(days=1)
                    
                    if start_date >= end_date:
                        print("Start date must be before end date")
                        continue
                    
                    end_date = min(end_date, datetime.now() - timedelta(days=1))
                    break
                except ValueError:
                    print("Invalid date format. Use YYYY-MM-DD")

            print("\nRunning backtest...")
            cumulative_returns, portfolio_returns = backtest_portfolio(
                final_weights, valid_tickers, start_date, end_date, boe_rate, etf_params, include_dividends, div_yields
            )
            
            if cumulative_returns is None:
                continue
                
            actual_start = cumulative_returns.first_valid_index()
            actual_end = cumulative_returns.last_valid_index()
            print(f"\nEffective backtest range: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
        
            print("\n=== BACKTEST RESULTS ===")
            print(f"Final Portfolio Value: {cumulative_returns.iloc[-1]*100:.2f}%")
            print(f"Annualized Return: {portfolio_returns.mean()*252*100:.2f}%")
            print(f"Annualized Volatility: {portfolio_returns.std()*np.sqrt(252)*100:.2f}%")
            print(f"Sharpe Ratio: {(portfolio_returns.mean()*252 - boe_rate)/(portfolio_returns.std()*np.sqrt(252)):.2f}")
            
            plot_backtest_results(cumulative_returns, portfolio_returns, boe_rate)

            backtest = input("\nWould you like to perform another backtest with different dates? (y/n): ").lower()


if __name__ == "__main__":
    main()