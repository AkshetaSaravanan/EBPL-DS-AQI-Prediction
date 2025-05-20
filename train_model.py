import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Paths
DATA_PATH = 'data/city_hour.csv'
SAVE_DIR_IMG = 'images/'
SAVE_DIR_MODEL = 'models/'

# Load and preprocess
df = pd.read_csv(DATA_PATH)
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df = df.dropna(subset=['Datetime'])
df = df[(df['Datetime'].dt.year >= 2020) & (df['Datetime'].dt.year <= 2024)]
df['Date'] = df['Datetime'].dt.date
df.to_csv('data/cleaned_data_2020_2024.csv', index=False)

# Group by City and Date
daily_df = df.groupby(['City', 'Date']).mean(numeric_only=True).reset_index()

# Train and Save Model per City
cities = ['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata']
for city in cities:
    city_df = daily_df[daily_df['City'] == city].copy()
    city_df['Date'] = pd.to_datetime(city_df['Date'])
    city_df['Date_ordinal'] = city_df['Date'].map(pd.Timestamp.toordinal)

    X = city_df[['Date_ordinal']]
    y = city_df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"{city}: RMSE = {mean_squared_error(y_test, y_pred):.2f}, R² = {r2_score(y_test, y_pred):.2f}")

    # Save model
    joblib.dump(model, f'{SAVE_DIR_MODEL}/{city}_model.pkl')

    # Save plot
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=city_df['Date'], y=city_df['AQI'], label='Actual AQI')
    plt.plot(city_df['Date'], model.predict(X), color='red', label='Regression Line')
    plt.title(f"{city} AQI Trend")
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR_IMG}/{city}_aqi_plot.png')
    plt.close()
