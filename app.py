import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

# Import backtesting function and data
from swing_trading_backtest import run_backtest, df, sequence_length, feature_cols, indicator_cols

# Initialize the Dash app
app = dash.Dash(__name__)

# Get available equities/indices from the data
available_equities = df['TckrSymb'].unique().tolist()

# Define the layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='Algo Trading Backtest Dashboard'),

    html.Div(children='Select an equity/index and run the backtest.'),

    # Equity/Index Selection
    html.Div([
        dcc.Dropdown(
            id='equity-dropdown',
            options=[{'label': i, 'value': i} for i in available_equities],
            value=available_equities[0] if available_equities else None,
            placeholder="Select an Equity/Index"
        ),
        html.Button('Run Backtest', id='run-backtest-button', n_clicks=0),
    ]),

    # Area to display backtest results and visualizations
    html.Div(id='backtest-results')
])

# Callback to run backtest and display results
@app.callback(
    Output('backtest-results', 'children'),
    [Input('run-backtest-button', 'n_clicks')],
    [State('equity-dropdown', 'value')]
)
def update_backtest_results(n_clicks, selected_equity):
    if n_clicks > 0 and selected_equity:
        # Filter data for the selected equity
        selected_equity_data = df[df['TckrSymb'] == selected_equity].copy()

        print("Date dtype:", selected_equity_data['date'].dtype)
        print("Sample values:", selected_equity_data['date'].head(10).tolist())

        # Convert to numeric if not already
        selected_equity_data['date'] = pd.to_numeric(selected_equity_data['date'], errors='coerce')

        # Parse as nanoseconds
        selected_equity_data['date'] = pd.to_datetime(selected_equity_data['date'], unit='ns', errors='coerce')

        # Drop rows where date could not be parsed
        selected_equity_data = selected_equity_data.dropna(subset=['date'])

        # Format for Plotly
        selected_equity_data['date'] = selected_equity_data['date'].dt.strftime('%Y-%m-%d')

        print("Parsed date values:", selected_equity_data['date'].head(10))

        # Run the backtest
        backtest_result = run_backtest(selected_equity, selected_equity_data, sequence_length, feature_cols, indicator_cols)

        # Display results
        if backtest_result:
            # Create Candlestick Chart
            candlestick_fig = go.Figure(data=[go.Candlestick(
                x=selected_equity_data['date'],
                open=selected_equity_data['open'],
                high=selected_equity_data['high'],
                low=selected_equity_data['low'],
                close=selected_equity_data['close'],
                name='Price'
            )])

            # Overlay Buy Trade Markers
            trade_buy_dates = [trade['buy_date'] for trade in backtest_result['trades']]
            trade_buy_prices = [trade['buy_price'] for trade in backtest_result['trades']]
            candlestick_fig.add_trace(go.Scatter(
                x=trade_buy_dates,
                y=trade_buy_prices,
                mode='markers',
                marker=dict(color='lime', size=14, symbol='triangle-up'),
                name='Buy Entry',
                hoverinfo='text',
                text=[f"Buy: {p:.2f}" for p in trade_buy_prices]
            ))

            # Overlay Sell Trade Markers
            trade_sell_dates = [trade['sell_date'] for trade in backtest_result['trades']]
            trade_sell_prices = [trade['sell_price'] for trade in backtest_result['trades']]
            candlestick_fig.add_trace(go.Scatter(
                x=trade_sell_dates,
                y=trade_sell_prices,
                mode='markers',
                marker=dict(color='red', size=14, symbol='triangle-down'),
                name='Sell Exit',
                hoverinfo='text',
                text=[f"Sell: {p:.2f}" for p in trade_sell_prices]
            ))

            # Layout: dark theme, zoom slider, grid, legend
            candlestick_fig.update_layout(
                template='plotly_dark',
                title=f'{selected_equity} Candlestick Chart with Trade Entries/Exits',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type='category',  # Use 'date' if your x is datetime
                    tickformat='%Y-%m-%d'
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Extract buy and sell signals and dates
            buy_signals = backtest_result['buy_signals'][sequence_length:]
            sell_signals = backtest_result['sell_signals'][sequence_length:]
            dates = selected_equity_data['date'].iloc[sequence_length:].tolist()

            # Add Buy Signal Markers
            buy_signal_dates = [dates[i] for i, signal in enumerate(buy_signals) if signal == 1]
            buy_signal_prices = [selected_equity_data['close'].iloc[sequence_length + i] for i, signal in enumerate(buy_signals) if signal == 1]

            candlestick_fig.add_trace(go.Scatter(
                x=buy_signal_dates,
                y=buy_signal_prices,
                mode='markers',
                marker=dict(color='blue', size=8, symbol='circle'),
                name='Buy Signal (Strategy)'
            ))

            # Add Sell Signal Markers
            sell_signal_dates = [dates[i] for i, signal in enumerate(sell_signals) if signal == 1]
            sell_signal_prices = [selected_equity_data['close'].iloc[sequence_length + i] for i, signal in enumerate(sell_signals) if signal == 1]

            candlestick_fig.add_trace(go.Scatter(
                x=sell_signal_dates,
                y=sell_signal_prices,
                mode='markers',
                marker=dict(color='orange', size=8, symbol='circle'),
                name='Sell Signal (Strategy)'
            ))

            # Extract buy and sell dates and prices from trades
            trade_buy_dates = [trade['buy_date'] for trade in backtest_result['trades']]
            trade_buy_prices = [trade['buy_price'] for trade in backtest_result['trades']]
            trade_sell_dates = [trade['sell_date'] for trade in backtest_result['trades']]
            trade_sell_prices = [trade['sell_price'] for trade in backtest_result['trades']]

            # Add Buy Trade Markers
            candlestick_fig.add_trace(go.Scatter(
                x=trade_buy_dates,
                y=trade_buy_prices,
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Buy Trade'
            ))

            # Add Sell Trade Markers
            candlestick_fig.add_trace(go.Scatter(
                x=trade_sell_dates,
                y=trade_sell_prices,
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Sell Trade'
            ))

            candlestick_fig.update_layout(title=f'{selected_equity} Candlestick Chart with Signals and Trade Markers',
                                          xaxis_title='Date',
                                          yaxis_title='Price')

            # Create Equity Curve Chart
            dates = selected_equity_data['date'].iloc[sequence_length:].astype(str).tolist()
            equity_curve = backtest_result['equity_curve'][:len(dates)]
            equity_curve_fig = go.Figure(data=[go.Scatter(
                x=dates,
                y=equity_curve,
                mode='lines',
                name='Equity Curve'
            )])
            equity_curve_fig.update_layout(title=f'{selected_equity} Equity Curve',
                                           xaxis_title='Date',
                                           yaxis_title='Equity Value')

            # Create Drawdown Chart
            drawdown = backtest_result['drawdown'][:len(dates)]
            drawdown_fig = go.Figure(data=[go.Scatter(
                x=dates,
                y=drawdown,
                mode='lines',
                fill='tozeroy',
                marker_color='red',
                name='Drawdown'
            )])
            drawdown_fig.update_layout(title=f'{selected_equity} Drawdown',
                                       xaxis_title='Date',
                                       yaxis_title='Drawdown (%)')

            # Create Trade Log Table
            trade_log_table = html.Table([
                html.Thead(html.Tr([html.Th(col) for col in ['Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'P&L']])),
                html.Tbody([
                    html.Tr([
                        html.Td(trade['buy_date']),
                        html.Td(f"{trade['buy_price']:.2f}"),
                        html.Td(trade['sell_date']),
                        html.Td(f"{trade['sell_price']:.2f}"),
                        html.Td(f"{trade['pnl']:.2f}"),
                    ]) for trade in backtest_result['trades']
                ])
            ])

            return html.Div([
                html.H3('Backtest Results'),
                dcc.Tabs([
                    dcc.Tab(label='Overview', children=[
                        html.Div([
                            html.P(f"Invested Capital: {backtest_result['equity_curve'][0]:.2f}"),
                            html.P(f"Total P&L: {backtest_result['total_pnl']:.2f}"),
                            html.P(f"Max Drawdown: {backtest_result['max_drawdown']:.2f}"),
                            html.P(f"Total Trades: {backtest_result['total_trades']}"),
                            html.P(f"Profitable Trades: {backtest_result['profitable_trades']}"),
                            html.P(f"Profit Factor: {backtest_result['profit_factor']:.2f}"),
                            html.P(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}" if backtest_result['sharpe_ratio'] is not None else "Sharpe Ratio: N/A"),
                            html.P(f"Sortino Ratio: {backtest_result['sortino_ratio']:.2f}" if backtest_result['sortino_ratio'] is not None else "Sortino Ratio: N/A"),
                        ])
                    ]),
                    dcc.Tab(label='Performance', children=[
                        dcc.Graph(figure=candlestick_fig),
                        dcc.Graph(figure=equity_curve_fig),
                        dcc.Graph(figure=drawdown_fig),
                    ]),
                    dcc.Tab(label='Trades Analysis', children=[
                        html.H4('Trade Log'),
                        trade_log_table
                    ]),
                ])
            ])
        else:
            return html.Div(f"Insufficient data to run backtest for {selected_equity}.")
    return html.Div() # Return empty div initially

if __name__ == '__main__':
    app.run(debug=True)
