import pandas as pd
from backtesting.framework.optimizer import Optimizer
from backtesting.framework.backtest_engine import BacktestEngine
from backtesting.framework.strategy_engine import SwingTradingStrategy

class WalkForward:

    def __init__(self, stock_data, symbol, initial_capital=200000):
        self.stock_data = stock_data
        self.symbol = symbol
        self.initial_capital = initial_capital

    def walkforward_run(self, train_years=3, test_years=1):
        results = []

        stock = self.stock_data[self.stock_data['TckrSymb'] == self.symbol].copy()
        stock = stock.sort_values(by="date").reset_index(drop=True)

        start_date = stock['date'].min()
        end_date = stock['date'].max()

        train_start = start_date

        while True:
            train_end = train_start + pd.DateOffset(years=train_years)
            test_end = train_end + pd.DateOffset(years=test_years)

            train_data = stock[(stock['date'] >= train_start) & (stock['date'] <= train_end)]
            test_data = stock[(stock['date'] > train_end) & (stock['date'] <= test_end)]

            if len(train_data) < 252 * train_years or len(test_data) < 252 * test_years:
                break  # not enough data left

            print(f"Running window: Train {train_start.date()} to {train_end.date()}, Test {train_end.date()} to {test_end.date()}")

            # Optimize on train data
            optimizer = Optimizer(train_data, self.symbol, initial_capital=self.initial_capital)
            opt_results = optimizer.run_grid_search()

            if opt_results.empty:
                print("No valid optimization found for train period.")
                break

            best_params = opt_results.iloc[0]
            risk_pct = best_params['risk_pct']
            reward_ratio = best_params['reward_ratio']
            max_holding_days = best_params['max_holding_days']

            # Run on test data
            strategy = SwingTradingStrategy()
            backtest = BacktestEngine(
                stock_data=test_data,
                symbol=self.symbol,
                initial_capital=self.initial_capital,
                risk_pct=risk_pct,
                reward_ratio=reward_ratio,
                max_holding_days=max_holding_days
            )
            test_result = backtest.run_backtest()

            if test_result:
                test_result.update({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end
                })
                results.append(test_result)

            # Move window forward
            train_start = train_start + pd.DateOffset(years=test_years)

        return pd.DataFrame(results)
