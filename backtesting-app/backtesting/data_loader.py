import pandas as pd

def load_data():
    """Load and prepare data for backtesting"""
    dtype_dict = {
        'TckrSymb': str,
        'date': str,
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32'
    }

    chunk_size = 10000
    all_chunks = []

    for chunk in pd.read_csv("/Users/avishekroy/Algo Trading/git repo/algo-trading/backtesting-app/historical_stock_data.csv",
                             usecols=dtype_dict.keys(),
                             dtype=dtype_dict,
                             chunksize=chunk_size):
        all_chunks.append(chunk)

    df = pd.concat(all_chunks)
    
    # Date parsing (same as your original)
    dates_ns = pd.to_datetime(df['date'], unit='ns', errors='coerce')
    failed_ns_mask = dates_ns.isna()
    dates_ddmmyyyy = pd.to_datetime(df['date'][failed_ns_mask], format='%d-%b-%Y', errors='coerce')
    df['date'] = dates_ns.combine_first(dates_ddmmyyyy)
    df.sort_values(by=["TckrSymb", "date"], inplace=True)
    
    return df

# Load data at module level for app.py compatibility
df = load_data()
