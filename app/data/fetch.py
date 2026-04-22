import yfinance as yf
import sqlite3
import pandas as pd
from pathlib import Path


DB_PATH = "app/data/forex.db"

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDINR": "USDINR=X"
}
PERIOD = "2y"     
INTERVAL = "1h"   

def fetch_and_store():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    for pair_name, ticker in PAIRS.items():
        print(f"Fetching {pair_name}...")
        try:
            df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
            
            if df.empty:
                print(f"No data found for {pair_name}.")
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel('Ticker')
                
            df.reset_index(inplace=True)
            
            df.to_sql(pair_name, conn, if_exists='replace', index=False)
            print(f"Stored {len(df)} rows for {pair_name}.")
            
        except Exception as e:
            print(f"Error fetching {pair_name}: {e}")

    conn.close()
    print("\nData pipeline complete! Your SQLite database is ready.")

if __name__ == "__main__":
    fetch_and_store()