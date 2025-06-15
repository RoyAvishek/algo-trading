from dash import dcc, html, dash_table
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from plotly.subplots import make_subplots

# Import styles
from styles import custom_styles

def create_metric_card(title, value, metric_type='neutral'):
    """Create an enhanced metric card component"""
    color_style = custom_styles['metric_value']
    if metric_type == 'positive':
        color_style = {**color_style, **custom_styles['positive_metric']}
    elif metric_type == 'negative':
        color_style = {**color_style, **custom_styles['negative_metric']}
    
    return html.Div(
        style={
            **custom_styles['metric_card'],
            ':hover': {'transform': 'translateY(-2px)'}
        }, 
        children=[
            html.Div(title, style=custom_styles['metric_title']),
            html.Div(value, style=color_style)
        ]
    )

def create_tradingview_chart(stock_data, trades, symbol):
    """Create an advanced TradingView-style chart with professional appearance"""
    
    # Create subplot structure
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{symbol} - Professional Trading Chart',
            'Volume Profile',
            'Technical Indicators'
        ),
        row_heights=[0.6, 0.25, 0.15],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ]
    )
    
    # === MAIN PRICE CHART ===
    # Add enhanced candlesticks
    fig.add_trace(
        go.Candlestick(
            x=stock_data['date'],
            open=stock_data['open'],
            high=stock_data['high'],
            low=stock_data['low'],
            close=stock_data['close'],
            name='OHLC',
            increasing=dict(
                line=dict(color='#089981', width=1),
                fillcolor='#089981'
            ),
            decreasing=dict(
                line=dict(color='#f23645', width=1),
                fillcolor='#f23645'
            ),
            hovertext=(
                '<b>%{x}</b><br>' +
                'Open: ₹%{open:.2f}<br>' +
                'High: ₹%{high:.2f}<br>' +
                'Low: ₹%{low:.2f}<br>' +
                'Close: ₹%{close:.2f}<br>' +
                '<extra></extra>'
            )
        ),
        row=1, col=1
    )
    
    # Add enhanced moving averages
    if len(stock_data) >= 200:
        stock_data_enhanced = stock_data.copy()
        stock_data_enhanced['EMA9'] = stock_data_enhanced['close'].ewm(span=9).mean()
        stock_data_enhanced['EMA21'] = stock_data_enhanced['close'].ewm(span=21).mean()
        stock_data_enhanced['EMA50'] = stock_data_enhanced['close'].ewm(span=50).mean()
        stock_data_enhanced['SMA200'] = stock_data_enhanced['close'].rolling(window=200).mean()
        
        # EMA 9 (Fast)
        fig.add_trace(
            go.Scatter(
                x=stock_data_enhanced['date'],
                y=stock_data_enhanced['EMA50'],
                name='EMA 50',
                line=dict(color='#ffeb3b', width=2),
                opacity=0.7,
                hovertemplate='EMA 50: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # SMA 200 (Trend)
        fig.add_trace(
            go.Scatter(
                x=stock_data_enhanced['date'],
                y=stock_data_enhanced['EMA21'],
                name='EMA 21',
                line=dict(color='#ff6b35', width=1.5),
                opacity=0.8,
                hovertemplate='EMA 21: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # EMA 50 (Slow)
        fig.add_trace(
            go.Scatter(
                x=stock_data_enhanced['date'],
                y=stock_data_enhanced['EMA50'],
                name='EMA 50',
                line=dict(color='#ffeb3b', width=2),
                opacity=0.7,
                hovertemplate='EMA 50: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # SMA 200 (Trend)
        fig.add_trace(
            go.Scatter(
                x=stock_data_enhanced['date'],
                y=stock_data_enhanced['SMA200'],
                name='SMA 200',
                line=dict(color='#9c27b0', width=2.5),
                opacity=0.6,
                hovertemplate='SMA 200: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # === VOLUME CHART ===
    # Enhanced volume bars with color coding
    volume_colors = []
    for i in range(len(stock_data)):
        if stock_data['close'].iloc[i] >= stock_data['open'].iloc[i]:
            volume_colors.append('#089981')
        else:
            volume_colors.append('#f23645')
    
    fig.add_trace(
        go.Bar(
            x=stock_data['date'],
            y=stock_data['volume'],
            name='Volume',
            marker=dict(color=volume_colors, opacity=0.6),
            hovertemplate='Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add volume moving average
    if len(stock_data) >= 20:
        volume_ma = stock_data['volume'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=stock_data['date'],
                y=volume_ma,
                name='Vol MA 20',
                line=dict(color='#ffc107', width=2),
                opacity=0.8,
                hovertemplate='Vol MA 20: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # === TECHNICAL INDICATORS ===
    if len(stock_data) >= 14:
        # Calculate RSI
        delta = stock_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=stock_data['date'],
                y=rsi,
                name='RSI',
                line=dict(color='#9c27b0', width=2),
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="#f23645", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#089981", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#868993", opacity=0.3, row=3, col=1)
    
    # === TRADE SIGNALS ===
    if trades and len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        # Enhanced Buy Signals
        for i, trade in enumerate(trades):
            # Buy markers
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_date']],
                    y=[trade['entry_price']],
                    mode='markers+text',
                    marker=dict(
                        color='#089981',
                        size=15,
                        symbol='triangle-up',
                        line=dict(width=3, color='#ffffff')
                    ),
                    text=['BUY'],
                    textposition='top center',
                    textfont=dict(color='white', size=10, family='Arial Black'),
                    name='Buy Entry' if i == 0 else '',
                    showlegend=True if i == 0 else False,
                    hovertemplate=(
                        '<b>BUY ENTRY</b><br>' +
                        f'Date: {trade["entry_date"].strftime("%Y-%m-%d")}<br>' +
                        f'Price: ₹{trade["entry_price"]:.2f}<br>' +
                        f'Quantity: {int(trade["position_size"])}<br>' +
                        f'Investment: ₹{trade["entry_price"] * trade["position_size"]:,.0f}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=1, col=1
            )
            
            # Sell markers
            fig.add_trace(
                go.Scatter(
                    x=[trade['exit_date']],
                    y=[trade['exit_price']],
                    mode='markers+text',
                    marker=dict(
                        color='#f23645',
                        size=15,
                        symbol='triangle-down',
                        line=dict(width=3, color='#ffffff')
                    ),
                    text=['SELL'],
                    textposition='bottom center',
                    textfont=dict(color='white', size=10, family='Arial Black'),
                    name='Sell Exit' if i == 0 else '',
                    showlegend=True if i == 0 else False,
                    hovertemplate=(
                        '<b>SELL EXIT</b><br>' +
                        f'Date: {trade["exit_date"].strftime("%Y-%m-%d")}<br>' +
                        f'Price: ₹{trade["exit_price"]:.2f}<br>' +
                        f'P&L: ₹{trade["pnl"]:,.0f}<br>' +
                        f'Exit Reason: {trade["exit_reason"]}<br>' +
                        f'Holding: {trade["holding_days"]} days<br>' +
                        '<extra></extra>'
                    )
                ),
                row=1, col=1
            )
            
            # Trade connection lines
            pnl_color = '#089981' if trade['pnl'] > 0 else '#f23645'
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_date'], trade['exit_date']],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    line=dict(color=pnl_color, width=2, dash='dot'),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
    
    # === SUPPORT & RESISTANCE LEVELS ===
    if len(stock_data) > 100:
        # Calculate key levels
        recent_high = stock_data['high'].rolling(window=50).max().iloc[-1]
        recent_low = stock_data['low'].rolling(window=50).min().iloc[-1]
        
        # Add support/resistance lines
        fig.add_hline(
            y=recent_high,
            line_dash="longdash",
            line_color="#ff6b35",
            opacity=0.6,
            annotation_text="Resistance",
            annotation_position="bottom right",
            annotation_font_color="#ff6b35",
            row=1, col=1
        )
        
        fig.add_hline(
            y=recent_low,
            line_dash="longdash", 
            line_color="#2962ff",
            opacity=0.6,
            annotation_text="Support",
            annotation_position="top right",
            annotation_font_color="#2962ff",
            row=1, col=1
        )
    
    # === LAYOUT CONFIGURATION ===
    fig.update_layout(
        # Main title
        title={
            'text': f'{symbol} - Professional Trading Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#d1d4dc', 'family': 'Arial'}
        },
        
        # Overall layout
        height=900,
        template=None,  # Custom template
        
        # Background colors
        plot_bgcolor='#0d1421',
        paper_bgcolor='#0d1421',
        
        # Font configuration
        font=dict(color='#d1d4dc', family='Arial, sans-serif', size=12),
        
        # Margins
        margin=dict(l=60, r=60, t=80, b=60),
        
        # Legend
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 34, 45, 0.8)',
            bordercolor='#434651',
            borderwidth=1,
            font=dict(color='#d1d4dc', size=11)
        ),
        
        # Hover configuration
        hoverlabel=dict(
            bgcolor='rgba(30, 34, 45, 0.95)',
            font_color='#d1d4dc',
            bordercolor='#434651',
            font_size=12,
            font_family='Arial'
        ),
        
        # Crossfilter cursor
        hovermode='x unified',
        
        # Modebar configuration
        modebar=dict(
            remove=['zoom2d', 'select2d']
        ),
        
        # Range selector for main chart
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=30, label="1M", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(count=180, label="6M", step="day", stepmode="backward"),
                    dict(count=365, label="1Y", step="day", stepmode="backward"),
                    dict(step="all", label="ALL")
                ],
                bgcolor='rgba(30, 34, 45, 0.8)',
                bordercolor='#434651',
                borderwidth=1,
                font=dict(color='#d1d4dc')
            ),
            rangeslider=dict(
                visible=True,
                thickness=0.06,
                bgcolor='rgba(30, 34, 45, 0.8)',
                bordercolor='#434651',
                borderwidth=1
            ),
        )
    )
    
    # Configure axes for all subplots
    # Main chart (Price)
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        fixedrange=False,
        gridcolor='rgba(67, 70, 81, 0.3)',
        showline=True,
        linewidth=1,
        linecolor='#434651',
        showspikes=True,
        spikecolor='#434651',
        spikethickness=1,
        spikedash='solid',
        row=1, col=1
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(67, 70, 81, 0.3)',
        showline=True,
        linewidth=1,
        linecolor='#434651',
        side='right',
        tickformat='.2f',
        tickprefix='₹',
        showspikes=True,
        spikecolor='#434651',
        spikethickness=1,
        row=1, col=1
    )

    # Volume chart
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(67, 70, 81, 0.3)',
        showline=True,
        linecolor='#434651',
        row=2, col=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(67, 70, 81, 0.3)',
        showline=True,
        linecolor='#434651',
        side='right',
        tickformat='.2s',
        row=2, col=1
    )
    
    # RSI chart
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(67, 70, 81, 0.3)',
        showline=True,
        linecolor='#434651',
        row=3, col=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(67, 70, 81, 0.3)',
        showline=True,
        linecolor='#434651',
        side='right',
        range=[0, 100],
        tickvals=[0, 30, 50, 70, 100],
        row=3, col=1
    )
    
    return fig

def create_performance_table(result):
    """Create enhanced performance metrics table"""
    # Calculate additional metrics
    total_trades = result['total_trades']
    winning_trades = result['profitable_trades']
    losing_trades = total_trades - winning_trades
    
    metrics_data = [
        {'Category': 'Capital', 'Metric': 'Initial Capital', 'Value': f"₹{200000:,.0f}"},
        {'Category': 'Capital', 'Metric': 'Final Capital', 'Value': f"₹{200000 + result['total_pnl']:,.0f}"},
        {'Category': 'Returns', 'Metric': 'Total P&L', 'Value': f"₹{result['total_pnl']:,.0f}"},
        {'Category': 'Returns', 'Metric': 'Return %', 'Value': f"{result['total_return_pct']:.2f}%"},
        {'Category': 'Returns', 'Metric': 'Annualized Return', 'Value': f"{result.get('annualized_return', 0):.2f}%"},
        {'Category': 'Trading', 'Metric': 'Total Trades', 'Value': str(total_trades)},
        {'Category': 'Trading', 'Metric': 'Winning Trades', 'Value': str(winning_trades)},
        {'Category': 'Trading', 'Metric': 'Losing Trades', 'Value': str(losing_trades)},
        {'Category': 'Trading', 'Metric': 'Win Rate', 'Value': f"{result['win_rate']:.1%}"},
        {'Category': 'Risk', 'Metric': 'Profit Factor', 'Value': f"{result['profit_factor']:.2f}"},
        {'Category': 'Risk', 'Metric': 'Max Drawdown', 'Value': f"{result['max_drawdown']:.2%}"},
        {'Category': 'Risk', 'Metric': 'Sharpe Ratio', 'Value': f"{result['sharpe_ratio']:.2f}"},
        {'Category': 'Trade Stats', 'Metric': 'Avg Win', 'Value': f"₹{result['avg_win']:,.0f}"},
        {'Category': 'Trade Stats', 'Metric': 'Avg Loss', 'Value': f"₹{result['avg_loss']:,.0f}"},
        {'Category': 'Trade Stats', 'Metric': 'Avg Holding Period', 'Value': f"{result['avg_holding_days']:.0f} days"},
    ]
    
    return dash_table.DataTable(
        data=metrics_data,
        columns=[
            {'name': 'Category', 'id': 'Category'},
            {'name': 'Metric', 'id': 'Metric'},
            {'name': 'Value', 'id': 'Value'}
        ],
        style_cell={
            'backgroundColor': '#1e222d',
            'color': '#d1d4dc',
            'textAlign': 'left',
            'padding': '12px',
            'border': '1px solid #2a2e39',
            'fontSize': '13px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': '#2962ff',
            'color': '#ffffff',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#252932'
            },
            {
                'if': {'filter_query': '{Category} = "Capital"'},
                'backgroundColor': '#1a2332',
            },
            {
                'if': {'filter_query': '{Category} = "Returns"'},
                'backgroundColor': '#1a3223',
            },
            {
                'if': {'filter_query': '{Category} = "Risk"'},
                'backgroundColor': '#32231a',
            },
            {
                'if': {'filter_query': '{Category} = "Trading"'},
                'backgroundColor': '#1a2332',
            },
            {
                'if': {'filter_query': '{Category} = "Trade Stats"'},
                'backgroundColor': '#2a1a32',
            },
        ],
        style_table={
            'overflowX': 'auto',
            'border': '1px solid #2a2e39',
            'borderRadius': '8px'
        },
        page_size=15
    )

def create_equity_curve(result):
    """Create enhanced equity curve with advanced analytics"""
    if 'equity_curve' not in result or not result['equity_curve']:
        return go.Figure()
    
    equity_data = pd.DataFrame(result['equity_curve'])
    
    # Create subplot structure
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            'Portfolio Equity Curve',
            'Drawdown Analysis',
            'Monthly Returns Heatmap'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # === EQUITY CURVE ===
    fig.add_trace(
        go.Scatter(
            x=equity_data['date'],
            y=equity_data['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2962ff', width=3),
            fill='tonexty',
            fillcolor='rgba(41, 98, 255, 0.1)',
            hovertemplate='Date: %{x}<br>Portfolio: ₹%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add benchmark line (initial capital)
    initial_capital = 200000
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="#868993",
        opacity=0.5,
        annotation_text="Initial Capital",
        annotation_position="bottom right",
        row=1, col=1
    )
    
    # === DRAWDOWN CHART ===
    if 'drawdown' in equity_data.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_data['date'],
                y=equity_data['drawdown'] * 100,
                mode='lines',
                name='Drawdown %',
                line=dict(color='#f23645', width=2),
                fill='tozeroy',
                fillcolor='rgba(242, 54, 69, 0.2)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
    
    # === MONTHLY RETURNS HEATMAP ===
    if len(equity_data) > 30:
        # Calculate monthly returns
        equity_data['date'] = pd.to_datetime(equity_data['date'])
        equity_data.set_index('date', inplace=True)
        monthly_returns = equity_data['equity'].resample('M').last().pct_change() * 100
        
        # Create heatmap data
        monthly_data = []
        years = sorted(monthly_returns.index.year.unique())
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        heatmap_data = []
        for year in years:
            year_data = []
            for month in range(1, 13):
                try:
                    value = monthly_returns[
                        (monthly_returns.index.year == year) & 
                        (monthly_returns.index.month == month)
                    ].iloc[0]
                    year_data.append(value)
                except:
                    year_data.append(None)
            heatmap_data.append(year_data)
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=months,
                y=years,
                colorscale=[
                    [0, '#f23645'],    # Red for losses
                    [0.5, '#ffffff'],  # White for neutral
                    [1, '#089981']     # Green for gains
                ],
                zmid=0,
                text=[[f'{val:.1f}%' if val is not None else '' for val in row] for row in heatmap_data],
                texttemplate='%{text}',
                textfont=dict(size=10),
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>',
                name='Monthly Returns'
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        plot_bgcolor='#0d1421',
        paper_bgcolor='#0d1421',
        font=dict(color='#d1d4dc', family='Arial'),
        title={
            'text': 'Portfolio Performance Analytics',
            'x': 0.5,
            'font': {'size': 18, 'color': '#d1d4dc'}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 34, 45, 0.8)',
            font=dict(color='#d1d4dc')
        )
    )
    
    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(67, 70, 81, 0.3)',
            showline=True,
            linecolor='#434651',
            row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(67, 70, 81, 0.3)',
            showline=True,
            linecolor='#434651',
            row=i, col=1
        )
    
    # Format y-axis for equity curve
    fig.update_yaxes(
        tickformat=',.0f',
        tickprefix='₹',
        row=1, col=1
    )
    
    # Format y-axis for drawdown
    fig.update_yaxes(
        tickformat='.1f',
        ticksuffix='%',
        row=2, col=1
    )
    
    return fig

def create_trades_table(trades):
    """Create enhanced trades history table"""
    if not trades:
        return html.Div("No trades executed", style={'textAlign': 'center', 'color': '#868993'})
    
    trades_data = []
    for i, trade in enumerate(trades, 1):
        trades_data.append({
            'Trade #': i,
            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
            'Entry Price': f"₹{trade['entry_price']:.2f}",
            'Exit Price': f"₹{trade['exit_price']:.2f}",
            'Quantity': int(trade['position_size']),
            'P&L': f"₹{trade['pnl']:,.0f}",
            'P&L Value': trade['pnl'], # Add raw P&L value for filtering
            'Return %': f"{((trade['exit_price'] / trade['entry_price']) - 1) * 100:.2f}%",
            'Holding Days': trade['holding_days'],
            'Exit Reason': trade['exit_reason']
        })
    
    return dash_table.DataTable(
        data=trades_data,
        columns=[
            {'name': col, 'id': col, 'hideable': True} if col == 'P&L Value' else {'name': col, 'id': col} for col in trades_data[0].keys()
        ],
        style_cell={
            'backgroundColor': '#1e222d',
            'color': '#d1d4dc',
            'textAlign': 'center',
            'padding': '10px',
            'border': '1px solid #2a2e39',
            'fontSize': '12px',
            'fontFamily': 'Arial',
            'whiteSpace': 'nowrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': '120px'
        },
        style_header={
            'backgroundColor': '#2962ff',
            'color': '#ffffff',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#252932'
            },
            {
                'if': {
                    'filter_query': '{P&L Value} < 0', # Filter based on numerical P&L Value
                    'column_id': 'P&L'
                },
                'color': '#f23645',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{P&L Value} >= 0', # Filter based on numerical P&L Value
                    'column_id': 'P&L'
                },
                'color': '#089981',
                'fontWeight': 'bold'
            }
        ],
        style_table={
            'overflowX': 'auto',
            'border': '1px solid #2a2e39',
            'borderRadius': '8px'
        },
        sort_action="native",
        filter_action="native",
        page_size=10
    )
