from openpyxl import load_workbook
import pandas as pd

EXCEL_PATH = "stocks.xlsx"


def save_stock(df: pd.DataFrame, symbol: str):
    df = df.copy()

    # Ensure Date is date-only for Excel
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Add returns if not present
    if "Returns" not in df.columns:
        df["Returns"] = df["Close"].pct_change()

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name=symbol, index=False)



def update_sheet_list():
    wb = load_workbook(EXCEL_PATH)

    stock_sheets = [
        s for s in wb.sheetnames
        if not s.startswith("__") and s != "Dashboard"
    ]

    stock_sheets.sort()

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace"
    ) as writer:
        pd.DataFrame(stock_sheets, columns=["Symbol"]) \
            .to_excel(writer, sheet_name="__SHEETS__", index=False)

    try:
        wb.save(EXCEL_PATH)
    except PermissionError:
        print("⚠ Close the Excel file before running the script.")
