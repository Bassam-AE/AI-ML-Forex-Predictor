import yfinance as yf

ticker = "EURUSD=X"
data = yf.download(ticker, period="2y", interval="1h")

data.to_csv("conversions/eurusd_2y_hourly.csv")
print("Data downloaded successfully!")