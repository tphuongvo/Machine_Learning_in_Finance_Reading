import yfinance as yf
import datetime
import numpy as np
import pandas as pd

def loading_stock(symbol ,start_time, end_time, interval_time): 

    """
    Input:  1. Stock information like the symbol 
            2. The range of time we want to scrap: start time, end time and the interval
                **Note that** the Yahoo! finance API 1m data is only retrievable for the last 7 days,
                and anything intraday (interval <1d) only for the last 60 days  

    Output: Two dataset:
            df: the raw loaded data
            stock_df: the raw data of which "Datetime" column has been converted to "Timestamp"
    """
    
    df = yf.download(tickers = symbol, interval = interval_time ,start=start_time,end=end_time)

    stock_df = df.copy()
    stock_df = stock_df.reset_index()

    # Convert Datetime
    stock_df["Datetime"] = stock_df["Datetime"].astype(str).str.rsplit("-",1, expand=True)[0]
    stock_df["Timestamp"] = stock_df["Datetime"].map(lambda t: datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
    stock_df.drop(columns = "Datetime", inplace = True)
    stock_df = stock_df.set_index("Timestamp")
    
    return df, stock_df

def add_data(input_df):    
    input_df.reset_index(drop = False, inplace= True)
    list_cols = input_df.columns.tolist()
    df_new = pd.DataFrame(columns= list_cols)
    df_new = pd.concat([df_new,input_df],ignore_index=True)
    
    return df_new

def scraping_stock(symbol ,start_time, end_time, interval_time):
    range_days = round(((end_time - start_time).days+1)/7)
    start = start_time
    for list in range(range_days):
        end = start + datetime.timedelta(days=6) 
        print(f'{list}  - Date : {start}')
        df_raw, stock_df_raw = loading_stock(symbol ,start, end, interval_time)
        
        df = add_data(df_raw)
        stock_df = add_data(stock_df_raw)
        start = start + datetime.timedelta(days=6)      
    return df, stock_df


def timetofloat(times):
    # Convert to the UTC+07:00
    t_float = times.timestamp() - 3600*7
    return t_float

def floatotime(t_float):    
    f_times = datetime.datetime.fromtimestamp(t_float).strftime("%Y-%m-%d %H:%M:%S")   
    return datetime.datetime.strptime(f_times, "%Y-%m-%d %H:%M:%S") 