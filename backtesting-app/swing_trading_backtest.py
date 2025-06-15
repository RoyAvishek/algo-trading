import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the strategy and data loader
from backtesting.strategy import SwingTradingStrategy
from backtesting.data_loader import load_data, df

# === MAIN EXECUTION ===
if __name__ == '__main__':
    # Test with multiple symbols for better results
    symbols_to_backtest = ['TCS', 'INFY', 'HDFCBANK', 'RELIANCE', 'ITC']  # Add more symbols
    
    strategy = SwingTradingStrategy(initial_capital=200000)
    backtest_results = []

    # Filter data for selected symbols
    df_filtered = df[df['TckrSymb'].isin(symbols_to_backtest)]

    # Run backtests
    for symbol in symbols_to_backtest:
        # Pass parameters directly to run_enhanced_backtest
        result = strategy.run_enhanced_backtest(symbol, df_filtered, risk_pct=2, reward_ratio=3, max_holding_days=60)
        if result:
            backtest_results.append(result)

    # Display results
    print("\n" + "="*80)
    print("ENHANCED SWING TRADING STRATEGY RESULTS")
    print("="*80)
    
    total_capital = 0
    total_trades = 0
    
    for result in backtest_results:
        print(f"\n📈 {result['symbol']}")
        print(f"   Total Return: ₹{result['total_pnl']:,.0f} ({result['total_return_pct']:.1f}%)")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Win Rate: {result['win_rate']:.1%}")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print(f"   Max Drawdown: {result['max_drawdown']:.1%}")
        print(f"   Avg Holding: {result['avg_holding_days']:.0f} days")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        
        total_capital += result['total_pnl']
        total_trades += result['total_trades']
    
    print(f"\n🎯 PORTFOLIO SUMMARY:")
    print(f"   Total P&L: ₹{total_capital:,.0f}")
    print(f"   Total Return: {(total_capital/200000)*100:.1f}%")
    print(f"   Total Trades: {total_trades}")
    print(f"   Avg Trades per Stock: {total_trades/len(backtest_results):.1f}")
    
    # Create visualization plots
    plt.style.use('seaborn-v0_8-darkgrid') # Updated style
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Portfolio Equity Curve
    ax1 = plt.subplot(2, 2, 1)
    for result in backtest_results:
        normalized_equity = [eq/result['equity_curve'][0] for eq in result['equity_curve']]
        ax1.plot(normalized_equity, label=result['symbol'])
    ax1.set_title('Portfolio Equity Curves')
    ax1.set_ylabel('Growth of ₹1')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Drawdown Analysis
    ax2 = plt.subplot(2, 2, 2)
    for result in backtest_results:
        drawdown = [eq/max(result['equity_curve'][:i+1]) - 1 for i, eq in enumerate(result['equity_curve'])]
        ax2.plot(drawdown, label=result['symbol'])
    ax2.set_title('Drawdown Analysis')
    ax2.set_ylabel('Drawdown %')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Return Distribution
    ax3 = plt.subplot(2, 2, 3)
    all_returns = []
    for result in backtest_results:
        equity_series = pd.Series(result['equity_curve'])
        returns = equity_series.pct_change().dropna()
        all_returns.extend(returns)
    ax3.hist(all_returns, bins=50, density=True, alpha=0.7)
    ax3.set_title('Return Distribution')
    ax3.set_xlabel('Daily Returns')
    ax3.set_ylabel('Frequency')
    
    # 4. Performance Metrics Comparison
    ax4 = plt.subplot(2, 2, 4)
    metrics = {
        'Win Rate': [result['win_rate'] for result in backtest_results],
        'Sharpe': [result['sharpe_ratio'] for result in backtest_results],
        'Max DD': [-result['max_drawdown'] for result in backtest_results]
    }
    
    x = np.arange(len(symbols_to_backtest))
    width = 0.25
    
    ax4.bar(x - width, metrics['Win Rate'], width, label='Win Rate')
    ax4.bar(x, metrics['Sharpe'], width, label='Sharpe')
    ax4.bar(x + width, metrics['Max DD'], width, label='Max DD')
    
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(symbols_to_backtest)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('backtest_analysis.png')
    
    # Additional Statistics
    print("\n📊 DETAILED STATISTICS:")
    print("="*80)
    
    # Calculate portfolio-level statistics
    portfolio_daily_returns = []
    for result in backtest_results:
        equity_series = pd.Series(result['equity_curve'])
        returns = equity_series.pct_change().dropna()
        portfolio_daily_returns.extend(returns)
    
    portfolio_daily_returns = np.array(portfolio_daily_returns)
    
    # Calculate additional metrics
    annual_return = np.mean(portfolio_daily_returns) * 252
    annual_volatility = np.std(portfolio_daily_returns) * np.sqrt(252)
    sortino_ratio = np.sqrt(252) * np.mean(portfolio_daily_returns) / np.std(portfolio_daily_returns[portfolio_daily_returns < 0]) if np.std(portfolio_daily_returns[portfolio_daily_returns < 0]) != 0 else 0
    
    print(f"\n📈 Portfolio Statistics:")
    print(f"   Annualized Return: {annual_return:.1%}")
    print(f"   Annualized Volatility: {annual_volatility:.1%}")
    print(f"   Sortino Ratio: {sortino_ratio:.2f}")
    
    # Calculate correlation matrix
    returns_by_symbol = {}
    for result in backtest_results:
        equity_series = pd.Series(result['equity_curve'])
        returns_by_symbol[result['symbol']] = equity_series.pct_change().dropna()
    
    returns_df = pd.DataFrame(returns_by_symbol)
    correlation_matrix = returns_df.corr()
    
    print("\n📊 Correlation Matrix:")
    print(correlation_matrix.round(2))
    
    # Save results to CSV
    results_df = pd.DataFrame([{
        'Symbol': result['symbol'],
        'Total Return (%)': result['total_return_pct'],
        'Total Trades': result['total_trades'],
        'Win Rate (%)': result['win_rate'] * 100,
        'Profit Factor': result['profit_factor'],
        'Max Drawdown (%)': result['max_drawdown'] * 100,
        'Avg Holding Days': result['avg_holding_days'],
        'Sharpe Ratio': result['sharpe_ratio']
    } for result in backtest_results])
    
    results_df.to_csv('backtest_results.csv', index=False)
    print("\n✅ Results saved to 'backtest_results.csv'")
