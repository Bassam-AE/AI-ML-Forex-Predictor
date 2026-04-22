import sqlite3
import pandas as pd

DB_PATH = "app/data/forex.db"
PAIRS = ["EURUSD", "GBPUSD", "USDINR"]
DAYS_TO_TEST = 10


def split_train_test():
    print(f"Connecting to {DB_PATH}...\n")
    conn = sqlite3.connect(DB_PATH)

    for pair in PAIRS:
        df = pd.read_sql(f'SELECT * FROM "{pair}"', conn)
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

        max_date = df["Datetime"].max()
        cutoff_date = max_date - pd.Timedelta(days=DAYS_TO_TEST)

        train_df = df[df["Datetime"] < cutoff_date].copy()
        test_df = df[df["Datetime"] >= cutoff_date].copy()

        train_df["Datetime"] = train_df["Datetime"].astype(str)
        test_df["Datetime"] = test_df["Datetime"].astype(str)

        # Write to separate tables — never touch the original source table
        train_df.to_sql(f"{pair}_TRAIN", conn, if_exists="replace", index=False)
        test_df.to_sql(f"{pair}_TEST", conn, if_exists="replace", index=False)

        print(f"{pair}:")
        print(f"   Training set : {len(train_df)} rows  (written to {pair}_TRAIN)")
        print(f"   Holdout set  : {len(test_df)} rows  (written to {pair}_TEST, cutoff {cutoff_date.date()})")

    conn.close()
    print("\nSplit complete. Original tables untouched.")


if __name__ == "__main__":
    split_train_test()
