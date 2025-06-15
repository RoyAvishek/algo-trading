import pandas as pd
import numpy as np

# Import the strategy and data
from backtesting.framework.strategy_engine import SwingTradingStrategy
from backtesting.framework.backtest_engine import BacktestEngine
from backtesting.data_loader import df

def run_backtest(symbol, stock_data, risk_pct, reward_ratio, max_holding_days):
    """
    Wrapper function to run backtest using the framework components.
    """
    # Use the new StrategyEngine to generate signals
    strategy_engine = SwingTradingStrategy()
    stock_with_signals = strategy_engine.generate_signals(stock_data[stock_data['TckrSymb'] == symbol].copy())

    if stock_with_signals is None or len(stock_with_signals) == 0:
        return None

    # Define configuration for the BacktestEngine
    config = {
        "risk_per_trade": risk_pct / 100.0,
        "reward_ratio": reward_ratio,
        "max_holding_days": max_holding_days,
        "trailing_stop_activation": 1.5 # Assuming a default for now
    }

    # Run the backtest using the new BacktestEngine
    backtest_engine = BacktestEngine(config)
    result = backtest_engine.run(symbol, stock_with_signals)

    if result is None:
        return None

    # Calculate additional metrics for UI compatibility
    equity_series = pd.Series([e['equity'] for e in result['equity_curve']], index=[e['date'] for e in result['equity_curve']]).astype(float)
    max_drawdown = ((equity_series - equity_series.cummax()) / equity_series.cummax()).min() if not equity_series.empty else 0
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    drawdown_series = ((equity_series - equity_series.cummax()) / equity_series.cummax()).tolist() if not equity_series.empty else []

    # Convert the result format to be compatible with the existing UI
    return {
        'symbol': result['symbol'],
        'total_pnl': result['total_pnl'],
        'max_drawdown': abs(max_drawdown), # Ensure drawdown is positive for display
        'total_trades': result['total_trades'],
        'profitable_trades': result['winning_trades'], # Map winning_trades to profitable_trades
        'profit_factor': result['profit_factor'],
        'equity_curve': result['equity_curve'],
        'trades': result['trades'],
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': 0, # Sortino not calculated by BacktestEngine
        'win_rate': result['win_rate'],
        'avg_win': result['avg_win'],
        'avg_loss': result['avg_loss'],
        'avg_holding_days': result['avg_holding_days'],
        'total_return_pct': result['total_return_pct'],
        'drawdown': drawdown_series # Provide calculated drawdown series
    }
