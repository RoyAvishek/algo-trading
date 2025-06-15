import numpy as np
import pandas as pd

class BacktestEngine:
    def __init__(self, config, initial_capital=200000):
        self.config = config
        self.initial_capital = initial_capital

    def calculate_position_size(self, entry_price, stop_loss):
        risk_amount = self.initial_capital * self.config['risk_per_trade']
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share > 0:
            return int(risk_amount / risk_per_share)
        return 0

    def run(self, symbol, stock):
        capital = self.initial_capital
        position = 0
        position_size = 0
        entry_price = 0
        stop_loss = 0
        target_price = 0
        trailing_stop = 0
        entry_date = None
        trades = []
        equity_curve = [{'date': stock['date'].iloc[0], 'equity': capital}]
        
        for i in range(len(stock)):
            row = stock.iloc[i]
            current_price = row['close']
            current_date = row['date']
            
            # ENTRY
            if row['entry_signal'] == 1 and position == 0:
                entry_price = current_price
                atr = row['ATR']
                stop_loss = entry_price - (2 * atr)
                risk = entry_price - stop_loss
                target_price = entry_price + (risk * self.config['reward_ratio'])
                position_size = self.calculate_position_size(entry_price, stop_loss)
                
                if position_size > 0:
                    position = position_size
                    capital -= position_size * entry_price
                    entry_date = current_date
                    trailing_stop = stop_loss

            # EXIT
            elif position > 0:
                if current_price >= entry_price + (entry_price - stop_loss) * self.config['trailing_stop_activation']:
                    new_trailing_stop = current_price - (2 * row['ATR'])
                    trailing_stop = max(trailing_stop, new_trailing_stop)
                
                exit_signal = False
                exit_reason = ""
                exit_price = current_price

                if current_price <= trailing_stop:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                    exit_price = trailing_stop
                elif current_price >= target_price:
                    exit_signal = True
                    exit_reason = "Target"
                    exit_price = target_price
                elif (current_date - entry_date).days >= self.config['max_holding_days']:
                    exit_signal = True
                    exit_reason = "Max Holding"
                    exit_price = current_price

                if exit_signal:
                    pnl = position * (exit_price - entry_price)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'holding_days': (current_date - entry_date).days
                    })
                    position, entry_price, stop_loss, target_price, trailing_stop, entry_date = 0, 0, 0, 0, 0, None
                    capital += pnl + (position_size * entry_price)
            
            # Equity Update
            unrealized_pnl = position * (current_price - entry_price) if position > 0 else 0
            equity_curve.append({'date': current_date, 'equity': capital + unrealized_pnl})

        # Closing final open position
        if position > 0:
            final_price = stock['close'].iloc[-1]
            final_date = stock['date'].iloc[-1]
            pnl = position * (final_price - entry_price)
            capital += position * final_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': final_date,
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_size': position,
                'pnl': pnl,
                'exit_reason': "Final Exit",
                'holding_days': (final_date - entry_date).days
            })

        if not trades:
            return None

        total_pnl = sum([t['pnl'] for t in trades])
        win_trades = [t for t in trades if t['pnl'] > 0]
        lose_trades = [t for t in trades if t['pnl'] < 0]

        return {
            'symbol': symbol,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / self.initial_capital) * 100,
            'total_trades': len(trades),
            'winning_trades': len(win_trades),
            'losing_trades': len(lose_trades),
            'win_rate': len(win_trades) / len(trades),
            'avg_win': np.mean([t['pnl'] for t in win_trades]) if win_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in lose_trades]) if lose_trades else 0,
            'profit_factor': (sum([t['pnl'] for t in win_trades]) / abs(sum([t['pnl'] for t in lose_trades]))) if lose_trades else float('inf'),
            'trades': trades,
            'avg_holding_days': np.mean([t['holding_days'] for t in trades]),
            'equity_curve': equity_curve
        }
