import pandas as pd

def load_gold_price_data(file_name):
    usecols = ["Name", "US dollar", "Euro", "Japanese yen", "Pound sterling", "Canadian dollar", "Swiss franc", "Indian rupee", "Chinese renmimbi"]
    df = pd.read_excel(file_name, sheet_name = "Daily", header = 8, index_col = 0, usecols = usecols)
    df = df.reset_index()
    df = df.rename(columns = {"Name": "Date"})
    #df["Date"] = pd.to_datetime(df["Date"])
    return df

def gold_price_xlsx2csv(xlsx_file_name, csv_file_name):
    df = load_gold_price_data(xlsx_file_name)
    df.to_csv(csv_file_name, index = False)

if __name__ == "__main__":
    print("Extracting gold prices from .xlsx, and converting to .csv...")
    gold_price_xlsx2csv("gold/Prices.xlsx", "gold/gold_price.csv")
    
