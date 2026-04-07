import pandas as pd
import xgboost as xgb
import numpy as np

# 1. Load data
df = pd.read_csv(r'C:\Users\Raghav\Desktop\Foursem\propagent\india_housing_prices.csv')

# 2. THE AUTO-INFER LOGIC (Case-Insensitive Keyword Search)
def get_col_name(keywords, columns):
    for col in columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    return None

# Mapping keywords to our standard names
price_col = get_col_name(['price', 'amount', 'lakh', 'cost'], df.columns)
bhk_col = get_col_name(['bhk', 'bedroom', 'room'], df.columns)
sqft_col = get_col_name(['sqft', 'area', 'size', 'feet'], df.columns)
year_col = get_col_name(['year', 'built', 'age', 'construction'], df.columns)

# Verify if we found them
if not all([price_col, bhk_col, sqft_col, year_col]):
    print(f"❌ Error: Could not find all columns automatically.")
    print(f"Found: Price={price_col}, BHK={bhk_col}, SqFt={sqft_col}, Year={year_col}")
    exit()

# 3. RENAME & NORMALIZE
df = df.rename(columns={
    price_col: 'price', 
    bhk_col: 'bhk', 
    sqft_col: 'sqft', 
    year_col: 'year_built'
})

# Convert Absolute Rupees to Lakhs automatically
if df['price'].mean() > 1000:
    df['price'] = df['price'] / 100000

# 4. TRAIN (With Monotone Constraints to fix the 2BHK vs 4BHK logic)
X = df[['bhk', 'sqft', 'year_built']]
y = df['price']

model = xgb.XGBRegressor(
    n_estimators=100, 
    monotone_constraints="(1, 1, 0)" # 1=Price must increase with BHK/SqFt
)
model.fit(X, y)

# 5. SAVE
model.save_model('prop_model.json')
print(f"✅ SUCCESS! Model trained using: {price_col}, {bhk_col}, {sqft_col}, {year_col}")