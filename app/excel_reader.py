import pandas as pd

def read_excel_file(file):
    if not file.name.endswith((".xlsx", ".xls")):
        raise ValueError("Please upload a valid Excel file.")
    xls = pd.ExcelFile(file)
    sheets_data = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    return sheets_data