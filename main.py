from data.data_loader import fetch_stock
from excel.excel_handler import save_stock, update_sheet_list
from plot.dashboard import run_console_dashboard


def main():                                                                            # Defining main function to ask for stock symbol 
    symbol = input("Enter stock symbol (e.g., RELIANCE.NS): ").strip().upper()          # Making input not be case sensitive   

    df = fetch_stock(symbol)                                                            # Fetch stock using yfinance (from data_loader.py)

    if df is None or df.empty:                                                          # Exception handling for invalid symbol
        print(f" Invalid symbol or no data found: {symbol}")
        return

    save_stock(df, symbol)                                                              # Save data in excel file (from excel_handler.py)
    update_sheet_list()                                                                 # Update the sheet list so we can choose between them 

    run_console_dashboard(df, symbol)                                                   # Run the main console for plotting



if __name__ == "__main__":
    main()