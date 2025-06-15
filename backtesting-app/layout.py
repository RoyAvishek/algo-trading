from dash import dcc, html, dash_table
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from plotly.subplots import make_subplots

# Import styles
from styles import custom_styles

# Import data for available equities
from backtesting.data_loader import df

# Get available equities/indices from the data
available_equities = sorted(df['TckrSymb'].unique().tolist())

# Define the enhanced layout
app_layout = html.Div(style=custom_styles['container'], children=[
    html.H1('📈 Backtesting Dashboard', style=custom_styles['header']),
    
    # Enhanced Control Panel
    html.Div(style=custom_styles['controls'], children=[
        html.Div([
            html.Label('Select Stock/Index:', 
                      style={'fontSize': '14px', 'marginBottom': '8px', 'display': 'block', 
                             'color': '#d1d4dc', 'fontWeight': '500'}),
            dcc.Dropdown(
                id='equity-dropdown',
                options=[{'label': symbol, 'value': symbol} for symbol in available_equities],
                value='TCS',
                placeholder='Select a stock...',
                style={
                    'backgroundColor': '#2a2e39',
                    'color': '#d1d4dc',
                    'border': '1px solid #434651'
                },
                clearable=False
            ),
        ], style={'marginBottom': '20px'}),
        
        # Strategy Parameters Section
        html.Div([
            html.Label('Strategy Parameters:', 
                      style={'fontSize': '16px', 'marginBottom': '15px', 'display': 'block',
                             'color': '#d1d4dc', 'fontWeight': '600'}),
            
            html.Div([
                # Risk per Trade
                html.Div([
                    html.Label('Risk per Trade (%):',
                              style={'fontSize': '12px', 'color': '#868993', 'marginBottom': '5px'}),
                    dcc.Input(
                        id='risk-input',
                        type='number',
                        value=2,
                        min=0.5,
                        max=5,
                        step=0.1,
                        style={
                            'width': '100px',
                            'backgroundColor': '#2a2e39',
                            'color': '#d1d4dc',
                            'border': '1px solid #434651',
                            'borderRadius': '4px',
                            'padding': '8px'
                        }
                    )
                ], style={'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
                
                # Reward Ratio
                html.Div([
                    html.Label('Reward Ratio:',
                              style={'fontSize': '12px', 'color': '#868993', 'marginBottom': '5px'}),
                    dcc.Input(
                        id='reward-input',
                        type='number',
                        value=3,
                        min=1,
                        max=10,
                        step=0.5,
                        style={
                            'width': '100px',
                            'backgroundColor': '#2a2e39',
                            'color': '#d1d4dc',
                            'border': '1px solid #434651',
                            'borderRadius': '4px',
                            'padding': '8px'
                        }
                    )
                ], style={'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
                
                # Max Holding Days
                html.Div([
                    html.Label('Max Holding Days:',
                              style={'fontSize': '12px', 'color': '#868993', 'marginBottom': '5px'}),
                    dcc.Input(
                        id='holding-input',
                        type='number',
                        value=60,
                        min=10,
                        max=120,
                        step=5,
                        style={
                            'width': '100px',
                            'backgroundColor': '#2a2e39',
                            'color': '#d1d4dc',
                            'border': '1px solid #434651',
                            'borderRadius': '4px',
                            'padding': '8px'
                        }
                    )
                ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
            ])
        ], style={'marginBottom': '25px'}),
        
        # Action Buttons
        html.Div([
            html.Button('🚀 Run Backtest', 
                       id='run-backtest-button', 
                       n_clicks=0, 
                       style={**custom_styles['button'], 'backgroundColor': '#2962ff'}),
            html.Button('📊 Compare Stocks', 
                       id='compare-button', 
                       n_clicks=0,
                       style={**custom_styles['button'], 'backgroundColor': '#089981'}),
            html.Button('📈 Technical Analysis', 
                       id='technical-button', 
                       n_clicks=0,
                       style={**custom_styles['button'], 'backgroundColor': '#ff6b35'}),
        ])
    ]),

    # Stock Comparison Input
    html.Div([
        html.Label('Enter Stock Symbols for Comparison (comma-separated):',
                   style={'fontSize': '14px', 'marginBottom': '8px', 'display': 'block',
                          'color': '#d1d4dc', 'fontWeight': '500'}),
        dcc.Input(
            id='comparison-symbols-input',
            type='text',
            placeholder='e.g., TCS, INFY, RELIANCE',
            style={
                'width': '100%',
                'backgroundColor': '#2a2e39',
                'color': '#d1d4dc',
                'border': '1px solid #434651',
                'borderRadius': '4px',
                'padding': '8px'
            }
        ),
    ], style={'marginBottom': '20px', 'maxWidth': '600px'}),

    # Enhanced Loading Indicator
    dcc.Loading(
        id="loading",
        type="cube",
        color="#2962ff",
        children=[html.Div(id='backtest-results')],
        style={'backgroundColor': 'transparent'}
    ),

    # Comparison Results Table
    html.Div(id='comparison-results-table-container', style={'marginTop': '30px'}),

    # Technical Analysis Output Container
    html.Div(id='technical-analysis-output', style={'marginTop': '30px'}),

    # Store components for data management
    dcc.Store(id='comparison-data'),
    dcc.Store(id='technical-data'),

    # Interval component for real-time updates (if needed)
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0,
        disabled=True
    )
])
