import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count

from backtesting.framework.strategy_engine import SwingTradingStrategy
from backtesting.framework.backtest_engine import BacktestEngine

class Optimizer:

    def __init__(self, stock_data, symbol, initial_capital=200000):
        self.stock_data = stock_data
        self.symbol = symbol
        self.initial_capital = initial_capital

    def run_single_backtest(self, params):
        risk_pct, reward_ratio, max_holding_days = params
        
        strategy = SwingTradingStrategy()
        backtest = BacktestEngine(
            stock_data=self.stock_data,
            symbol=self.symbol,
            initial_capital=self.initial_capital,
            risk_pct=risk_pct,
            reward_ratio=reward_ratio,
            max_holding_days=max_holding_days
        )

        result = backtest.run_backtest()

        if result is None:
            return None

        return {
            'risk_pct': risk_pct,
            'reward_ratio': reward_ratio,
            'max_holding_days': max_holding_days,
            'total_return_pct': result['total_return_pct'],
            'profit_factor': result['profit_factor'],
            'max_drawdown': result['max_drawdown'],
            'sharpe_ratio': result['sharpe_ratio'],
            'win_rate': result['win_rate'],
            'total_trades': result['total_trades'],
            'avg_holding_days': result['avg_holding_days']
        }

    def run_grid_search(self):
        # Define parameter grid (fully adjustable)
        risk_pct_range = [1, 1.5, 2, 2.5, 3]
        reward_ratio_range = [2, 2.5, 3, 3.5, 4]
        max_holding_days_range = [30, 60, 90, 120]

        param_grid = list(product(risk_pct_range, reward_ratio_range, max_holding_days_range))

        print(f"Running optimizer for {len(param_grid)} combinations...")

        with Pool(cpu_count()) as pool:
            results = pool.map(self.run_single_backtest, param_grid)

        # Filter out None results (if any)
        valid_results = [r for r in results if r is not None]
        return pd.DataFrame(valid_results).sort_values(by=['profit_factor', 'sharpe_ratio'], ascending=False)
