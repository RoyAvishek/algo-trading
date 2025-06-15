import dash
from dash import dcc, html, dash_table, callback
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from plotly.subplots import make_subplots

# Import layout and callbacks
from layout import app_layout
from callbacks import register_callbacks

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the app layout
app.layout = app_layout

# Register callbacks
register_callbacks(app)

# === RUN THE APP ===
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
