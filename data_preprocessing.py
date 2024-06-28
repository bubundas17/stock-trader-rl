import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['Returns'] = df['Close'].pct_change()
    return df
