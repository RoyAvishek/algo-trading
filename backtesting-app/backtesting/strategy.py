import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ADXIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
from ta.volume import ChaikinMoneyFlowIndicator
import warnings
warnings.filterwarnings('ignore')

class SwingTradingStrategy:
    def __init__(self, initial_capital=200000):
        self.initial_capital = initial_capital
        self.risk_per_trade = 0.02  # Fixed risk
        self.trailing_stop_activation = 1.5

    def calculate_position_size(self, entry_price, stop_loss):
        risk_amount = self.initial_capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        return int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

    def run_enhanced_backtest(self, symbol, stock_data, risk_pct, reward_ratio_unused, max_holding_days):
        print(f"Processing stock: {symbol}")
        stock = stock_data[stock_data['TckrSymb'] == symbol].copy()
        if len(stock) < 60:
            return None

        self.risk_per_trade = risk_pct / 100.0
        self.max_holding_days = max_holding_days

        stock = stock.sort_values('date').reset_index(drop=True)
        stock['RSI'] = RSIIndicator(close=stock['close'], window=14).rsi()
        stock['EMA20'] = EMAIndicator(close=stock['close'], window=20).ema_indicator()
        stock['EMA50'] = EMAIndicator(close=stock['close'], window=50).ema_indicator()
        stock['SMA200'] = SMAIndicator(close=stock['close'], window=200).sma_indicator()
        stock['ATR'] = AverageTrueRange(high=stock['high'], low=stock['low'], close=stock['close'], window=14).average_true_range()
        stock['CMF'] = ChaikinMoneyFlowIndicator(high=stock['high'], low=stock['low'], close=stock['close'], volume=stock['volume'], window=20).chaikin_money_flow()
        stock['ADX'] = ADXIndicator(high=stock['high'], low=stock['low'], close=stock['close'], window=14).adx()
        stock['volume_ma'] = stock['volume'].rolling(window=20).mean()

        stock['entry_signal'] = 0
        for i in range(len(stock)):
            if stock['close'].iloc[i] > stock['EMA20'].iloc[i] and stock['volume'].iloc[i] > stock['volume_ma'].iloc[i] * 1.2:
                if 30 < stock['RSI'].iloc[i] < 80 and stock['ADX'].iloc[i] > 20:
                    stock.loc[i, 'entry_signal'] = 1

        capital = self.initial_capital
        position = 0
        entry_price, stop_loss, target_price, trailing_stop = 0, 0, 0, 0
        entry_date = None
        trades = []
        equity_curve = [{'date': stock['date'].iloc[0], 'equity': capital}]

        for i in range(len(stock)):
            current_price = stock['close'].iloc[i]
            current_date = stock['date'].iloc[i]
            adx_value = stock['ADX'].iloc[i]

            if stock['entry_signal'].iloc[i] == 1 and position == 0:
                entry_price = current_price
                atr = stock['ATR'].iloc[i]
                stop_loss = entry_price - 2 * atr

                # Dynamic Reward:Risk logic
                if adx_value >= 30:
                    reward_ratio = 4.0
                elif adx_value >= 20:
                    reward_ratio = 3.0
                else:
                    reward_ratio = 2.0

                risk = entry_price - stop_loss
                target_price = entry_price + (risk * reward_ratio)
                position_size = self.calculate_position_size(entry_price, stop_loss)

                if position_size > 0:
                    position = position_size
                    capital -= position_size * entry_price
                    entry_date = current_date
                    trailing_stop = stop_loss

            elif position > 0:
                if current_price >= entry_price + (entry_price - stop_loss) * self.trailing_stop_activation:
                    new_trailing_stop = current_price - 2 * stock['ATR'].iloc[i]
                    trailing_stop = max(trailing_stop, new_trailing_stop)

                exit_signal = False
                exit_reason = ""
                exit_price = current_price

                if current_price <= trailing_stop:
                    exit_signal, exit_reason, exit_price = True, "Stop Loss", trailing_stop
                elif current_price >= target_price:
                    exit_signal, exit_reason, exit_price = True, "Target", target_price
                elif (current_date - entry_date).days >= self.max_holding_days:
                    exit_signal, exit_reason, exit_price = True, "Max Holding Days", current_price

                if exit_signal:
                    trades.append({
                        'entry_date': entry_date, 'exit_date': current_date,
                        'entry_price': entry_price, 'exit_price': exit_price,
                        'position_size': position, 'pnl': position * (exit_price - entry_price),
                        'exit_reason': exit_reason, 'holding_days': (current_date - entry_date).days
                    })
                    position, entry_price, stop_loss, target_price, trailing_stop, entry_date = 0, 0, 0, 0, 0, None

            equity_curve.append({'date': current_date, 'equity': capital + (position * current_price if position > 0 else 0)})

        if position > 0:
            final_price = stock['close'].iloc[-1]
            final_date = stock['date'].iloc[-1]
            trades.append({
                'entry_date': entry_date, 'exit_date': final_date,
                'entry_price': entry_price, 'exit_price': final_price,
                'position_size': position, 'pnl': position * (final_price - entry_price),
                'exit_reason': "Final Exit", 'holding_days': (final_date - entry_date).days
            })

        if not trades:
            return None

        total_pnl = sum([t['pnl'] for t in trades])
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] < 0]
        win_rate = len(win_trades) / len(trades)
        avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
        profit_factor = (abs(avg_win) * len(win_trades)) / (abs(avg_loss) * len(loss_trades)) if loss_trades else float('inf')
        equity_series = pd.Series([e['equity'] for e in equity_curve], index=[e['date'] for e in equity_curve]).astype(float)
        max_drawdown = ((equity_series - equity_series.cummax()) / equity_series.cummax()).min()
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0

        return {
            'symbol': symbol, 'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / self.initial_capital) * 100,
            'total_trades': len(trades), 'winning_trades': len(win_trades),
            'losing_trades': len(loss_trades), 'win_rate': win_rate,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor,
            'max_drawdown': abs(max_drawdown), 'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_curve, 'trades': trades,
            'avg_holding_days': np.mean([t['holding_days'] for t in trades])
        }
