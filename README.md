# NYC Green Taxi Fare Predictor

## Project Overview
This project analyzes NYC Green Taxi trip data from March 2024 and builds machine learning models to predict taxi fares based on trip characteristics. The repository includes both the data analysis/modeling code and a Streamlit web application for making fare predictions.



## Features
- **Data Analysis**: Comprehensive exploration of NYC Green Taxi trip data
- **Machine Learning Models**: Multiple models including Linear Regression and Gradient Boosting
- **Interactive Web App**: User-friendly Streamlit interface for fare prediction
- **Visualization**: Insightful charts and graphs showing trip patterns and model results

## Project Structure
- `nyc_green_taxi_analysis_march_2024.ipynb`: Data cleaning, exploration, and model development
- `app.py`: Streamlit application for fare prediction
- `requirements.txt`: Required Python packages
- Models (to be saved after running the analysis):
  - `multiple_linear_regression_model.pkl`: Linear Regression model
  - `best_model_gradient_boosting.pkl`: Gradient Boosting Regressor model (best performer)

## Data Analysis Highlights
The analysis revealed several interesting patterns:
- Credit Card is the dominant payment method (70.0%)
- Street-hail trips represent 98.4% of all taxi rides
- Friday is the busiest day for taxi trips
- Peak hours show clear commuting patterns with morning and evening spikes
- Trip distance and duration are the strongest predictors of fare amount

## Model Performance
After evaluating multiple regression models:
- **Best Model**: Gradient Boosting Regressor
- **R² Score**: 0.896 (cross-validated)
- **RMSE**: 2.20
- **MAE**: 1.55

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda for package management

### Setup Instructions
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nyc-taxi-fare-predictor.git
   cd nyc-taxi-fare-predictor
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the analysis script (this will take some time):
   ```
   python nyc_green_taxi_analysis_march_2024.ipynb
   ```

5. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```

## Using the Web App
The Streamlit web app provides an intuitive interface for predicting taxi fares:

1. Enter trip details:
   - Trip distance
   - Estimated duration
   - Pickup date and time
   - Number of passengers
   - Payment method
   - Trip type

2. Click "Predict Fare" to get an estimate
3. Review the prediction and feature importance visualization

## Data Source
The analysis uses NYC Green Taxi Trip data from the NYC Taxi & Limousine Commission (TLC), focusing on March 2024 data. The dataset includes information about pickup/dropoff times, locations, payment types, fare amounts, and other trip-related metrics.

## Key Findings
- Trip distance and duration are the strongest predictors of fare amount
- Cash payments show significantly lower tip amounts than credit card payments
- Weekend rides tend to have higher average fares than weekday rides
- Peak hour pricing strategies could be optimized based on demand patterns
- Trip types show different fare patterns, suggesting differentiated pricing strategies

## Future Work
- Analyze seasonal patterns by incorporating data from multiple months
- Explore geospatial analysis of pickup and dropoff locations
- Integrate weather data to understand impact on taxi demand and fares
- Develop a real-time fare prediction system based on the current model
- Investigate customer segmentation based on trip patterns and payment behavior

## Author
Pavithra Sevakula
