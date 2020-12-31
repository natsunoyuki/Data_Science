import pandas as pd
import yfinance

def download_stock_data(Stocks, start_date, end_date):
    """
    Wrapper function for yfinance.download()
    
    Inputs
    ------
    Stocks: list
        list of stocks to download from Yahoo finance
    start_date: datetime
        start date to download
    end_date: datetime
        end date to download
        
    Returns
    -------
    stocks: pd.DataFrame
        pd.DataFrame of stock data
    """
    # The downloaded stock data has the columns:
    # "Stock", "Open", "High", "Low", "Close", "Adj Close", "Volume"
    want = ["Stock", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    stocks = pd.DataFrame(columns = want)
    for i in Stocks:  
        print("Downloading:", i)
        stock = yfinance.download(i, start = start_date, end = end_date)
        # reset index to force the index to become "Date"
        stock = stock.reset_index()
        # add the stock's name into the data
        stock["Stock"] = i
        stocks = pd.concat([stocks, stock])  
    # re-arrange the columns as follows:
    want = ["Stock", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    stocks = stocks[want] 
    return stocks
