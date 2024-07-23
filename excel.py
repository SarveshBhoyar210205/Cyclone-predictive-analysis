import pandas as pd
df=pd.read_csv('h5.csv')
df.to_excel("h5.xlsx",sheet_name="Testing",index=False)