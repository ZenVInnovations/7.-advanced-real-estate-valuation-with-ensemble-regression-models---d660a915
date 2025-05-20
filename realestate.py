import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Excel file
file_path = 'data_real_estate_data.xlsx'
df = pd.read_excel(file_path)

# Strip any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Show cleaned column names
print("Cleaned Columns in Excel:", df.columns.tolist())

# Required columns
required_cols = ['HouseID', 'Area', 'Bedrooms', 'Bathroom', 'Price']
if not all(col in df.columns for col in required_cols):
    raise ValueError("One or more required columns are missing in the Excel sheet.")

# Features and target
X = df[['Area', 'Bedrooms', 'Bathroom']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Predict for specific house by HouseID
input_id = input("\nEnter HouseID to predict its value: ")

# Search for that house
house_row = df[df['HouseID'].astype(str).str.lower() == input_id.lower()]

if not house_row.empty:
    features = house_row[['Area', 'Bedrooms', 'Bathroom']]
    actual_price = house_row['Price'].values[0]
    predicted_price = model.predict(features)[0]

    print(f"\nDetails for HouseID '{input_id}':")
    print(f"Area: {features['Area'].values[0]}")
    print(f"Bedrooms: {features['Bedrooms'].values[0]}")
    print(f"Bathroom: {features['Bathroom'].values[0]}")
    print(f"Actual Price: {actual_price}")
    print(f"Predicted Price: {predicted_price:.2f}")
else:
    print(f"No house found with HouseID '{input_id}'. Please check the ID.")