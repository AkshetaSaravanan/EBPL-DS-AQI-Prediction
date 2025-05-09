# EBPL-DS: Predicting Air Quality Using Machine Learning (2020–2024)

This is Data Science project aimed at predicting AQI (Air Quality Index) for 5 major Indian cities using historical data and Linear Regression.

## About the Project

Air pollution is a rising concern across India. This project analyzes AQI trends from 2020 to 2024 and builds city-wise prediction models using clean and interpretable ML techniques. The goal is to help visualize how air quality has changed and forecast it for any future date.

## Objectives

- Analyze AQI levels for 5 cities: Bangalore, Delhi, Mumbai, Chennai, and Kolkata
- Clean the dataset and create daily city-wise averages
- Use Linear Regression to predict AQI based on date
- Categorize AQI values (Good, Moderate, Poor, etc.)
- Allow users to input a city + date and see predicted AQI

## Dataset Used

- Source: [Kaggle – City Hourly Air Quality Data](https://www.kaggle.com)
- File: `city_hour.csv` → Cleaned to `cleaned_data_2020_2024.csv`

## Tools & Technologies

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Visual Outputs

We created:
- 5 scatter plots (one for each city)
- 5 scatter + regression line plots (AQI predictions)
- All plots are in the `/Plots/` folder

## How to Run This Project

1. Clone this repo or download the ZIP
2. Open the notebook: `AQI_Prediction_2020_2024.ipynb`
3. Run all cells in Jupyter or Google Colab
4. Try out the user input block to get AQI prediction for any city + date

## Team Members

- **Lubika Palanisamy** – Data cleaning, model building, notebook development
- **Deeksha S** – EDA, feature engineering, AQI classification
- **Aksheta S** – Regression visualization, code formatting, testing

## Notebook Link

[Click here to open the notebook](your-notebook-link-here)

## License

This project is for academic and educational use only.
