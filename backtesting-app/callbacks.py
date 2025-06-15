from dash.dependencies import Input, Output, State
from dash import callback, html, dcc, dash_table
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# Import custom styles
from styles import custom_styles

# Import backtesting function and data
from backtesting.wrapper import run_backtest
from backtesting.data_loader import df

# Import UI component creation functions
from components import create_metric_card, create_tradingview_chart, create_performance_table, create_trades_table, create_equity_curve

def register_callbacks(app):
    """Register all callbacks with the Dash app instance."""

    @callback(
        Output('backtest-results', 'children'),
        [Input('run-backtest-button', 'n_clicks')],
        [State('equity-dropdown', 'value'),
         State('risk-input', 'value'),
         State('reward-input', 'value'),
         State('holding-input', 'value')]
    )
    def update_backtest_results(n_clicks, selected_equity, risk_pct, reward_ratio, max_holding_days):
        """Main callback for running backtest"""
        if n_clicks == 0:
            return html.Div([
                html.Div([
                    html.H3('🎯 Welcome to Professional Trading Dashboard', 
                           style={'textAlign': 'center', 'color': '#2962ff', 'marginBottom': '20px'}),
                    html.P('Select your preferred stock/index and configure strategy parameters above, then click "Run Backtest" to begin analysis.',
                           style={'textAlign': 'center', 'color': '#d1d4dc', 'fontSize': '16px', 'marginBottom': '30px'}),
                    
                    # Feature highlights
                    html.Div([
                        html.Div([
                            html.H4('📊 Advanced Analytics', style={'color': '#2962ff', 'marginBottom': '10px'}),
                            html.P('Professional-grade charting with TradingView-style interface, technical indicators, and comprehensive performance metrics.',
                                   style={'color': '#868993', 'fontSize': '14px'})
                        ], style={'backgroundColor': '#1e222d', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #2a2e39'}),
                        
                        html.Div([
                            html.H4('🎯 Risk Management', style={'color': '#089981', 'marginBottom': '10px'}),
                            html.P('Sophisticated position sizing, stop-loss management, and reward-to-risk ratio optimization for consistent profitability.',
                                   style={'color': '#868993', 'fontSize': '14px'})
                        ], style={'backgroundColor': '#1e222d', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #2a2e39'}),
                        
                        html.Div([
                            html.H4('📈 Real-time Insights', style={'color': '#ff6b35', 'marginBottom': '10px'}),
                            html.P('Interactive charts, trade visualization, equity curve analysis, and detailed performance breakdowns for informed decision making.',
                                   style={'color': '#868993', 'fontSize': '14px'})
                        ], style={'backgroundColor': '#1e222d', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #2a2e39'}),
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))', 'gap': '20px'})
                    
                ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '40px'})
            ])
        
        try:
            # Pass parameters from UI to backtest function
            result = run_backtest(
                symbol=selected_equity,
                stock_data=df,
                risk_pct=risk_pct,
                reward_ratio=reward_ratio,
                max_holding_days=max_holding_days
            )
            
            if not result or result['total_trades'] == 0:
                return html.Div([
                    html.Div([
                        html.H3('⚠️ No Trading Opportunities Found', 
                               style={'textAlign': 'center', 'color': '#ff6b35', 'marginBottom': '20px'}),
                        html.P(f'No valid swing trading setups were identified for {selected_equity} with the current parameters.',
                               style={'textAlign': 'center', 'color': '#d1d4dc', 'fontSize': '16px', 'marginBottom': '20px'}),
                        html.P('Try adjusting the strategy parameters or selecting a different stock/index.',
                               style={'textAlign': 'center', 'color': '#868993', 'fontSize': '14px'})
                    ], style={'backgroundColor': '#1e222d', 'padding': '40px', 'borderRadius': '8px', 'border': '1px solid #f23645', 'textAlign': 'center'})
                ])
            
            # Get stock data for charting
            stock_data = df[df['TckrSymb'] == selected_equity].copy()
            stock_data.columns = [col.lower() for col in stock_data.columns]
            stock_data = stock_data.rename(columns={'tckrsymb': 'symbol'})
            
            # Create enhanced results layout
            return html.Div([
                # Key Performance Metrics Cards
                html.Div([
                    create_metric_card(
                        'Total Return', 
                        f"₹{result['total_pnl']:,.0f}", 
                        'positive' if result['total_pnl'] > 0 else 'negative'
                    ),
                    create_metric_card(
                        'Return %', 
                        f"{result['total_return_pct']:.2f}%", 
                        'positive' if result['total_return_pct'] > 0 else 'negative'
                    ),
                    create_metric_card('Total Trades', str(result['total_trades'])),
                    create_metric_card(
                        'Win Rate', 
                        f"{result['win_rate']:.1%}", 
                        'positive' if result['win_rate'] > 0.5 else 'negative'
                    ),
                    create_metric_card(
                        'Profit Factor', 
                        f"{result['profit_factor']:.2f}", 
                        'positive' if result['profit_factor'] > 1 else 'negative'
                    ),
                    create_metric_card(
                        'Max Drawdown', 
                        f"{result['max_drawdown']:.2%}", 
                        'negative' if result['max_drawdown'] < -0.05 else 'neutral'
                    ),
                ], style=custom_styles['metrics_container']),
                
                # Enhanced Tabs Layout
                dcc.Tabs(
                    id='results-tabs',
                    value='chart-tab',
                    children=[
                        dcc.Tab(
                            label='📊 Price Chart & Signals',
                            value='chart-tab',
                            style=custom_styles['tab_style'],
                            selected_style=custom_styles['tab_selected_style']
                        ),
                        dcc.Tab(
                            label='📈 Portfolio Performance',
                            value='performance-tab',
                            style=custom_styles['tab_style'],
                            selected_style=custom_styles['tab_selected_style']
                        ),
                        dcc.Tab(
                            label='📋 Trade History',
                            value='trades-tab',
                            style=custom_styles['tab_style'],
                            selected_style=custom_styles['tab_selected_style']
                        ),
                        dcc.Tab(
                            label='📊 Analytics',
                            value='analytics-tab',
                            style=custom_styles['tab_style'],
                            selected_style=custom_styles['tab_selected_style']
                        )
                    ],
                    style={'marginBottom': '20px'}
                ),
                
                # Tab Content
                html.Div(id='tab-content', children=[
                    # Default: Chart Tab
                    html.Div([
                        dcc.Graph(
                            figure=create_tradingview_chart(stock_data, result['trades'], selected_equity),
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': f'{selected_equity}_trading_chart',
                                    'height': 900,
                                    'width': 1400,
                                    'scale': 2
                                }
                            },
                            style={'height': '900px'}
                        )
                    ])
                ])
            ])
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.H3('❌ Error Running Backtest', 
                           style={'textAlign': 'center', 'color': '#f23645', 'marginBottom': '20px'}),
                    html.P(f'An error occurred while processing the backtest: {str(e)}',
                           style={'textAlign': 'center', 'color': '#d1d4dc', 'fontSize': '16px', 'marginBottom': '20px'}),
                    html.P('Please check your parameters and try again.',
                           style={'textAlign': 'center', 'color': '#868993', 'fontSize': '14px'})
                ], style={'backgroundColor': '#1e222d', 'padding': '40px', 'borderRadius': '8px', 'border': '1px solid #f23645', 'textAlign': 'center'})
            ])

    @callback(
        Output('tab-content', 'children'),
        [Input('results-tabs', 'value')],
        [State('equity-dropdown', 'value'),
         State('risk-input', 'value'),
         State('reward-input', 'value'),
         State('holding-input', 'value')],
        prevent_initial_call=True
    )
    def update_tab_content(active_tab, selected_equity, risk_pct, reward_ratio, max_holding_days):
        """Update tab content based on selection"""
        if active_tab is None:
            return html.Div()
        
        try:
            # Pass parameters from UI to backtest function
            result = run_backtest(
                symbol=selected_equity,
                stock_data=df,
                risk_pct=risk_pct,
                reward_ratio=reward_ratio,
                max_holding_days=max_holding_days
            )
            
            if not result or result['total_trades'] == 0:
                return html.Div("No data available", style={'textAlign': 'center', 'color': '#868993'})
            
            stock_data = df[df['TckrSymb'] == selected_equity].copy()
            stock_data.columns = [col.lower() for col in stock_data.columns]
            stock_data = stock_data.rename(columns={'tckrsymb': 'symbol'})
            
            if active_tab == 'chart-tab':
                return html.Div([
                    dcc.Graph(
                        figure=create_tradingview_chart(stock_data, result['trades'], selected_equity),
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'{selected_equity}_trading_chart',
                                'height': 900,
                                'width': 1400,
                                'scale': 2
                            }
                        },
                        style={'height': '900px'}
                    )
                ])
            
            elif active_tab == 'performance-tab':
                return html.Div([
                    dcc.Graph(
                        figure=create_equity_curve(result),
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'{selected_equity}_performance',
                                'height': 800,
                                'width': 1200,
                                'scale': 2
                            }
                        },
                        style={'height': '800px'}
                    )
                ])
            
            elif active_tab == 'trades-tab':
                return html.Div([
                    html.H3('Trade History', 
                           style={'color': '#2962ff', 'marginBottom': '20px', 'textAlign': 'center'}),
                    create_trades_table(result['trades'])
                ])
            
            elif active_tab == 'analytics-tab':
                return html.Div([
                    html.H3('Performance Analytics', 
                           style={'color': '#2962ff', 'marginBottom': '20px', 'textAlign': 'center'}),
                    create_performance_table(result)
                ])
            
        except Exception as e:
            return html.Div(f"Error loading tab content: {str(e)}", 
                           style={'textAlign': 'center', 'color': '#f23645'})

    @callback(
        Output('technical-analysis-output', 'children'),
        [Input('technical-button', 'n_clicks')],
        [State('equity-dropdown', 'value')]
    )
    def display_technical_analysis(n_clicks, selected_equity):
        """Callback for displaying technical analysis"""
        if n_clicks == 0:
            return html.Div()

        if not selected_equity:
            return html.Div("Please select a stock/index.", style={'textAlign': 'center', 'color': '#868993'})

        try:
            # Load stock data
            stock_data = df[df['TckrSymb'] == selected_equity].copy()
            stock_data.columns = [col.lower() for col in stock_data.columns]
            stock_data = stock_data.rename(columns={'tckrsymb': 'symbol'})

            if stock_data.empty:
                return html.Div(f"No data found for {selected_equity}.", style={'textAlign': 'center', 'color': '#868993'})

            # Calculate a simple moving average (e.g., 20 periods)
            window = 20
            if len(stock_data) >= window:
                stock_data['SMA'] = stock_data['close'].rolling(window=window).mean()
            else:
                stock_data['SMA'] = np.nan # Not enough data points

            # Create a Plotly figure
            fig = go.Figure()

            # Add candlestick trace
            fig.add_trace(go.Candlestick(
                x=stock_data['date'],
                open=stock_data['open'],
                high=stock_data['high'],
                low=stock_data['low'],
                close=stock_data['close'],
                name='Price'
            ))

            # Add SMA trace if calculated
            if 'SMA' in stock_data.columns:
                 fig.add_trace(go.Scatter(
                    x=stock_data['date'],
                    y=stock_data['SMA'],
                    mode='lines',
                    name=f'{window}-Day SMA',
                    line=dict(color='#ff6b35', width=2)
                ))

            # Update layout for better appearance
            fig.update_layout(
                title=f'{selected_equity} Technical Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                plot_bgcolor='#1e222d',
                paper_bgcolor='#1e222d',
                font=dict(color='#d1d4dc'),
                hovermode='x unified'
            )

            return html.Div([
                html.H3(f'Technical Analysis for {selected_equity}',
                       style={'color': '#2962ff', 'marginBottom': '20px', 'textAlign': 'center'}),
                dcc.Graph(
                    figure=fig,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['select2d', 'zoom2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{selected_equity}_technical_analysis',
                            'height': 800,
                            'width': 1200,
                            'scale': 2
                        }
                    },
                    style={'height': '800px'}
                )
            ])

        except Exception as e:
            return html.Div(f"Error loading technical analysis: {str(e)}",
                           style={'textAlign': 'center', 'color': '#f23645'})


    @callback(
        Output('comparison-data', 'data'),
        [Input('compare-button', 'n_clicks')],
        [State('risk-input', 'value'),
         State('reward-input', 'value'),
         State('holding-input', 'value'),
         State('comparison-symbols-input', 'value')]
    )
    def run_comparison(n_clicks, risk_pct, reward_ratio, max_holding_days, comparison_symbols_input):
        """Run comparison across multiple stocks"""
        if n_clicks == 0:
            return {}

        if not comparison_symbols_input:
            return {}

        comparison_symbols = [symbol.strip() for symbol in comparison_symbols_input.split(',') if symbol.strip()]

        comparison_results = {}

        for symbol in comparison_symbols:
            try:
                result = run_backtest(
                    symbol=symbol,
                    stock_data=df,
                    risk_pct=risk_pct,
                    reward_ratio=reward_ratio,
                    max_holding_days=max_holding_days
                )
                
                if result and result['total_trades'] > 0:
                    comparison_results[symbol] = {
                        'Total P&L': result['total_pnl'],
                        'Total Return %': result['total_return_pct'],
                        'Total Trades': result['total_trades'],
                        'Winning Trades': result['profitable_trades'],
                        'Win Rate': result['win_rate'],
                        'Profit Factor': result['profit_factor'],
                        'Max Drawdown': result['max_drawdown'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Sortino Ratio': result.get('sortino_ratio', 0),
                        'Avg Win': result['avg_win'],
                        'Avg Loss': result['avg_loss'],
                        'Avg Holding Days': result['avg_holding_days']
                    }
            except Exception as e:
                print(f"Error running backtest for {symbol}: {e}")
                continue
        
        return comparison_results

    @callback(
        Output('comparison-results-table-container', 'children'),
        [Input('comparison-data', 'data')],
        prevent_initial_call=True
    )
    def display_comparison_results(comparison_data):
        """Display comparison results in a table"""
        if not comparison_data:
            return html.Div("No comparison data available.", style={'textAlign': 'center', 'color': '#868993'})

        # Convert dictionary to DataFrame
        df_comparison = pd.DataFrame.from_dict(comparison_data, orient='index')
        df_comparison = df_comparison.reset_index().rename(columns={'index': 'Stock'})

        # Define columns for the DataTable
        columns = [{"name": i, "id": i} for i in df_comparison.columns]

        # Format numerical columns to 2 decimal places
        for col in df_comparison.columns:
            if df_comparison[col].dtype in ['float64', 'float32']:
                df_comparison[col] = df_comparison[col].round(2)

        # Create and return the DataTable
        return html.Div([
            html.H3('Stock Comparison Results',
                    style={'color': '#2962ff', 'marginBottom': '20px', 'textAlign': 'center'}),
            dash_table.DataTable(
                id='comparison-table',
                columns=columns,
                data=df_comparison.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#1e222d',
                    'color': '#d1d4dc',
                    'border': '1px solid #2a2e39',
                    'padding': '10px',
                    'textAlign': 'center'
                },
                style_header={
                    'backgroundColor': '#2a2e39',
                    'color': '#d1d4dc',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#2a2e39'
                    }
                ],
            )
        ])
