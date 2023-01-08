import sys
sys.path.insert(1, '/path/to/ModelConfig/loading')
import loading as load

import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf


class AlgoStructure:

    # Calculate VWAP
    def vwap_cal(df):
        df.reset_index(drop = False, inplace = True)
        df["Typical_price"] = (df['Adj Close']+df['High']+df['Low'])*1/3
        price = df["Typical_price"]
        volume = df['Volume']
        # df['numerator'] = (price * volume).cumsum()
        # df['denominator'] = price.cumsum()
        df['Vwap'] = (price * volume).cumsum()/volume.cumsum()
        df.set_index('Timestamp', inplace = True)
        # df.assign(Vwap=(price * volume).cumsum() / price.cumsum())
        return df

    def cal_prchanges(price):
        return 100*(price.diff()/price)

    def cal_returns(price):
        log_returns = np.log(price/price.shift(1)).replace([np.inf, -np.inf], np.nan)
        return log_returns.dropna(), (pd.Series(log_returns, index=log_returns.index[1:]))

    # Time Bar chart    
    def timeplot(df, x_dim, y_dim):
        x = df[x_dim]
        y = df[y_dim]
        fig = px.line(df, x = x, y = y, title='Time bar of VWAP in 15 mins period')
        fig.update_traces(line=dict(color="Blue", width=1.5))
        fig.show()
    
    def volume_plot(df, x_dim, y_dim):
        x = df[x_dim]
        y = df[y_dim]
        fig =  px.bar(df, x=x, y=y, title = 'Trade Volume')
        fig.show()

    # Tick bar chart (as known as Candlesticks chart)

    def generate_tickbars(ticks, frequency):

        """
        Each array represent one trade
        Each trade is composed of: Time, price and quantity.
        However when it is ploted, Tick bars consider only one contract (volume) 
        for all the contracts each trader excecuted. 
        The information do not give us how many contracts each traders exchange
        in 1 minute, so we assume that each minute just fit 1 trader only.
        The frequency : a pre-defined number of transactions
        """        

        times = ticks['Timestamp']
        prices = ticks['Close']
        # volumes = ticks['Volume']
        res = np.zeros(shape=(len(range(0, len(ticks), frequency)), 6))
        it = 0
        for i in range(frequency, len(ticks), frequency):
            res[it][0] = load.timetofloat(times[i-1])           
            res[it][1] = prices[i-frequency]               
            res[it][2] = np.max(prices[i-frequency:i])     
            res[it][3] = np.min(prices[i-frequency:i])     
            res[it][4] = prices[i-1]                      
            # res[it][5] = np.sum(volumes[i-frequency:i])
            res[it][5] = len(range(i-frequency,i))   # Volume for exchange  
            it += 1
            
        return res

    # Volume Bar chart
    def generate_volumebars(trades, frequency):

        """
        Each array represent one transation
        Each transation is composed of: Time, price and quantity.
        Volumn bars grab all transactions operated by each trader. 
        """    
        
        times = trades['Timestamp']
        prices = trades['Close']
        volumes = trades['Volume']
        # lst_i = 0
        it = 0
        res = np.zeros(shape=(len(range(0, len(trades), frequency)), 6))

        for i in range(frequency, len(prices), frequency):
            res[it][0] = load.timetofloat(times[i-1])           
            res[it][1] = prices[i-frequency]              
            res[it][2] = np.max(prices[i-frequency:i])     
            res[it][3] = np.min(prices[i-frequency:i])    
            res[it][4] = prices[i-1]                       
            res[it][5] = np.sum(volumes[i-frequency:i])    
            # res[it][5] = np.sum(range(i-frequency,i))        
            it += 1

        return res

    # Statistical Properties
    
    # Check auto-correlation
    def get_test_stats(bar_types,bar_returns,test_func,*args,**kwds):
    
        dct = {bar:(int(bar_ret.shape[0]), test_func(bar_ret,*args,**kwds)) 
            for bar,bar_ret in zip(bar_types,bar_returns)}
        df = (pd.DataFrame.from_dict(dct)
            .rename(index={0:'sample_size',1:f'{test_func.__name__}_stat'})
            .T)
        return df

    def create_corr_plot(name,series, plot_pacf=False):
        corr_array = pacf(series.dropna(), alpha=0.05,nlags = 120) if plot_pacf else acf(series.dropna(), alpha=0.05,nlags = 120)
        lower_y = corr_array[1][:,0] - corr_array[0]
        upper_y = corr_array[1][:,1] - corr_array[0]

        fig = go.Figure()
        [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
        for x in range(len(corr_array[0]))]
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                    marker_size=8)
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
                fill='tonexty', line_color='rgba(255,255,255,0)')
        fig.update_traces(showlegend=False)
        fig.update_xaxes(range=[-1,42])
        fig.update_yaxes(zerolinecolor='#000000')
        
        title=f'{name} - Partial Autocorrelation (PACF)' if plot_pacf else f'{name} - Autocorrelation (ACF)'

        fig.update_layout(title = title,
            width=800,height=200,
            margin=dict(l=5, r=30, t=38, b=5)
            )
        fig.show()