import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import vectorbt as vbt
import csv
import numpy as np


def should_buy(current_buy, percent_over_high, percent_over_low):
    buy = False
    if not current_buy:
        if percent_over_low > 1.05:
            buy = True
        return buy

    buy = True
    if percent_over_high < 0.95:
        buy = False
    return buy


PERCENTAGE_FOR_NEW_HIGH_LOW_POINT = 0.02

data_path = "./raw_data/SPY.csv"

processed = []

infile = open(data_path)
reader = csv.reader(infile)
i = 0
# prev_close_low = 0
# current_close_low = 0
# prev_close_high = 0
# current_close_high = 0
is_trending_up = False
first_trend_found = False

low_closes = [0, 0, 0]
high_closes = [0, 0, 0]
consecutive_high_points = 0
consecutive_low_points = 0

next(reader)
for row in reader:
    date = row[0]
    open = round(float(row[1]), 2)
    high = round(float(row[2]), 2)
    low = round(float(row[3]), 2)
    close = round(float(row[4]), 2)
    volume = int(row[6])

    print(date)

    if i == 0:
        processed.append([date, open, high, low, close, volume, 0, 0, 0, 0, 0, 0, False])
        i += 1
        continue

    prev_close_high = processed[high_closes[1]][4]
    current_close_high = processed[high_closes[2]][4]
    prev_close_low = processed[low_closes[1]][4]
    current_close_low = processed[low_closes[2]][4]
    new_high_found = False
    new_low_found = False
    if not first_trend_found:
        if close / prev_close_high > (1 + PERCENTAGE_FOR_NEW_HIGH_LOW_POINT):
            is_trending_up = True
            first_trend_found = True
            high_closes[2] = i

        elif close / prev_close_low < (1 - PERCENTAGE_FOR_NEW_HIGH_LOW_POINT):
            is_trending_up = False
            first_trend_found = True
            low_closes[2] = i
    else:
        if is_trending_up:
            if close > current_close_high:
                high_closes[2] = i
            elif close < current_close_high and (current_close_high / close > (1 + PERCENTAGE_FOR_NEW_HIGH_LOW_POINT)):
                if current_close_high > processed[high_closes[1]][4]:
                    if consecutive_high_points < 0:
                        consecutive_high_points = 0
                    consecutive_high_points += 1
                elif current_close_high < processed[high_closes[1]][4]:
                    if consecutive_high_points > 0:
                        consecutive_high_points = 0
                    consecutive_high_points -= 1

                high_closes[0] = high_closes[1]
                high_closes[1] = high_closes[2]
                low_closes[2] = i
                is_trending_up = False
                new_high_found = True
                for j in range(high_closes[1], i):
                    processed[j][6] = processed[high_closes[1]][4]
                    processed[j][8] = processed[j][4] / processed[j][6]
                    processed[j][10] = consecutive_high_points
        else:
            if close < current_close_low:
                low_closes[2] = i
            elif close > current_close_low and (close / current_close_low > (1 + PERCENTAGE_FOR_NEW_HIGH_LOW_POINT)):
                if current_close_low < processed[low_closes[1]][4]:
                    if consecutive_low_points > 0:
                        consecutive_low_points = 0
                    consecutive_low_points -= 1
                elif current_close_low > processed[low_closes[1]][4]:
                    if consecutive_low_points < 0:
                        consecutive_low_points = 0
                    consecutive_low_points += 1
                low_closes[0] = low_closes[1]
                low_closes[1] = low_closes[2]
                high_closes[2] = i
                is_trending_up = True
                new_low_found = True
                for j in range(low_closes[1], i):
                    processed[j][7] = processed[low_closes[1]][4]
                    processed[j][9] = processed[j][4] / processed[j][7]
                    processed[j][11] = consecutive_low_points

    percent_over_high = close / current_close_high
    percent_over_low = close / current_close_low
    # sell = processed[i - 1][12] and not is_trending_up and current_close_high < prev_close_high
    buy = should_buy(processed[i - 1][12], percent_over_high, percent_over_low)
    print(f"Buy: {buy}")
    processed.append([date, open, high, low, close, volume, processed[high_closes[1]][4], processed[low_closes[1]][4], percent_over_high, percent_over_low, consecutive_high_points, consecutive_low_points, buy])

    i += 1

# processed_np = np.array(processed, dtype=[('Date', object), ('Open', np.float64), ('High', np.float64), ('Low', np.float64), ('Close', np.float64), ('Volume', np.int32)])
processed_np = np.array(processed, dtype=object)
# print(processed_np.dtype)
df = pd.DataFrame(processed_np, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Prev_High', 'Prev_Low', 'Percent_Over_High', 'Percent_Over_Low', 'Consecutive_High_Points', 'Consecutive_Low_Points', 'Buy_Signal'])
df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': int, 'Prev_High': float, 'Prev_Low': float, 'Percent_Over_High': float, 'Percent_Over_Low': float, 'Consecutive_High_Points': int, 'Consecutive_Low_Points': int, 'Buy_Signal': bool})
# print(df.dtypes)
# df = pd.read_csv(data_path, sep=",", parse_dates=True)
# print(df.dtypes)
# VWAP requires the DataFrame index to be a DatetimeIndex.
# Replace "datetime" with the appropriate column from your DataFrame
# df.set_index(pd.DatetimeIndex(df["Date"]), inplace=True)

# Calculate Returns and append to the df DataFrame
# df.ta.macd(append=True)

# df.ta.sma(length=20, append=True)
# df.ta.sma(length=50, append=True)
# df.ta.sma(length=100, append=True)

# df.columns

# Take a peek
# print(df.tail())

# df.ta.rsi(length=14, append=True)
# df.ta.sma(close="RSI", length=5, append=True, prefix="RSI")
# df.ta.sma(close="RSI", length=20, append=True, prefix="RSI")
# df.ta.ha(append=True)
# print(df.loc[df["Date"] == "2024-06-18"])
# df[["close", "EMA_10"]].tail(252).plot(figsize=(16,8), color=["black", "blue"], grid=True)
# plt.show()
# df["ENTRY_SIGNAL"] = (df["HA_close"] > df["HA_open"]) & (df["HA_low"] >= df["HA_open"])
# df["EXIT_SIGNAL"] = (df["HA_close"] < df["HA_open"]) & (df["HA_high"] <= df["HA_open"])


# def set_highs_lows(row):
#     global all_time_high_close
#     if row['Close'] > all_time_high_close:
#         return row['Close']
#     return all_time_high_close


# all_time_high_close = df.loc[6000, 'Close']
# df['All_Time_High_CLose'] = df.apply(set_highs_lows, axis=1)

# df.ta.sma(5, append=True)
# df.ta.sma(10, append=True)
# df["GC"] = df.ta.sma(close="RSI", length=20, append=True, prefix="RSI") > df.ta.sma(close="RSI", length=100, append=True, prefix="RSI")
# df["GC"] = df["SMA_50"] > df["SMA_200"]

df["BUY_SIGNAL"] = df["Buy_Signal"]

# Create boolean Signals(TS_Entries, TS_Exits) for vectorbt
# golden = df.ta.tsignals(df.GC, asbool=True, append=True)
golden = df.ta.tsignals(df.BUY_SIGNAL, asbool=True, append=True)

# print(df.loc[df["Date"] == "2023-12-05"])
# print(df.tail())
# df = df[6000:]
# # Create the Signals Portfolio
# pf = vbt.Portfolio.from_signals(df.Close, entries=df.ENTRY_SIGNAL, exits=df.EXIT_SIGNAL, freq="D", init_cash=10_000, fees=0.0, slippage=0.0)
pf = vbt.Portfolio.from_signals(df.Close, entries=golden.TS_Entries, exits=golden.TS_Exits, freq="D", init_cash=10_000, fees=0.0, slippage=0.0)
pf.trades.plot(title="Trades", height=500, width=1000).show()
# # Print Portfolio Stats and Return Stats
print(pf.stats())
# print(pf.returns_stats())
