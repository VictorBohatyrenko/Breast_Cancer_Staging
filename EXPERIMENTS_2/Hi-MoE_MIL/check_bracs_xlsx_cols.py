import pandas as pd
import sys
xlsx_path = sys.argv[1] if len(sys.argv) > 1 else "BRACS.xlsx"
df = pd.read_excel(xlsx_path, sheet_name='WSI_Information')
df.columns = [c.strip() for c in df.columns]
print("Колонки:", list(df.columns))
print(df.head(10).to_string())
