import pandas as pd

def convert_excel_to_csv(excel_path, csv_path, sheet_name=0):
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

# File paths
excel_path = 'h12345.xlsx'
csv_path = 'dataset_h12345.csv'

# Convert Excel to CSV
convert_excel_to_csv(excel_path, csv_path)
