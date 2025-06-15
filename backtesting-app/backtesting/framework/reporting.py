import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Reporting:

    def __init__(self, result):
        self.result = result

    def plot_equity_curve(self):
        equity = pd.DataFrame(self.result['equity_curve'])
        equity['date'] = pd.to_datetime(equity['date'])
        equity.set_index('date', inplace=True)
        plt.figure(figsize=(14,6))
        plt.plot(equity['equity'], label="Equity Curve", color="blue")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_drawdown(self):
        equity = pd.DataFrame(self.result['equity_curve'])
        equity['date'] = pd.to_datetime(equity['date'])
        equity.set_index('date', inplace=True)
        equity['max'] = equity['equity'].cummax()
        equity['drawdown'] = equity['equity'] / equity['max'] - 1

        plt.figure(figsize=(14,6))
        plt.fill_between(equity.index, equity['drawdown'], color='red', alpha=0.3)
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown %")
        plt.grid()
        plt.show()

    def print_trade_statistics(self):
        print("========== PERFORMANCE SUMMARY ==========")
        print(f"Symbol: {self.result['symbol']}")
        print(f"Total Return: {self.result['total_return_pct']:.2f}%")
        print(f"Total PnL: {self.result['total_pnl']:.2f}")
        print(f"Sharpe Ratio: {self.result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.result['max_drawdown']*100:.2f}%")
        print(f"Profit Factor: {self.result['profit_factor']:.2f}")
        print(f"Total Trades: {self.result['total_trades']}")
        print(f"Winning Trades: {self.result['winning_trades']}")
        print(f"Losing Trades: {self.result['losing_trades']}")
        print(f"Win Rate: {self.result['win_rate']*100:.2f}%")
        print(f"Avg Holding Days: {self.result['avg_holding_days']:.1f}")

    def show_trade_log(self):
        trades_df = pd.DataFrame(self.result['trades'])
        if trades_df.empty:
            print("No trades executed.")
            return

        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        print(trades_df)
