# PM2.5 Smart City Dashboard — Air Quality Prediction

This project predicts **PM2.5 (Particulate Matter ≤ 2.5 µm)** using geolocation and weather-based features and visualizes results through an interactive **Streamlit dashboard**.

---

## Features
- PM2.5 prediction from user-entered inputs
- Batch prediction using CSV upload
- Map-based visualization of predicted values (PyDeck)
- Display of model metrics (MAE, RMSE, feature importances)
- Downloadable predictions

---

## Input Features
| Feature | Description |
|--------|-------------|
| Latitude | Location |
| Longitude | Location |
| Temperature (°C) | Weather |
| Humidity (%) | Weather |
| Wind Speed (m/s) | Weather |
| Distance from city center | Spatial |
| Industrial zone flag | Spatial |
| Hour | Temporal |
| Day of week | Temporal |
| Month | Temporal |

---

## Running the App
```bash
pip install -r requirements.txt
streamlit run app.py
```
---

## How to Use the App

### Initial Setup
1. **Load a Model**: 
   - Upload your trained `.pkl` model file via the sidebar, OR
   - Click "Load default model" to use `air_quality_model.pkl`
   - **Note**: You'll also need a `scaler.pkl` file for single predictions

### Features

#### 1. Single Prediction Tab
- Enter individual values for latitude, longitude, temperature, humidity, and wind speed
- Click "Predict single point" to get PM2.5 prediction
- View the predicted location on an interactive map

#### 2. Batch CSV Prediction Tab
- Upload a CSV file with columns: `latitude`, `longitude`, `temperature`, `humidity`, `wind_speed`
- Optional: Include `pm25` column for ground truth to compute accuracy metrics (MAE, RMSE)
- Download a sample CSV template using the "Download sample CSV" button
- View all predictions on an interactive map
- Download results as CSV

#### 3. Model & Metrics Tab
- View model details and feature importances/coefficients
- Upload validation CSV (must include `pm25` column) to compute performance metrics
- Visualize prediction accuracy with scatter plots

### Input Data Format
Your CSV should have these columns:
```
latitude,longitude,temperature,humidity,wind_speed
28.61,77.20,30.0,50.0,3.0
```

For validation, add `pm25` column with ground truth values.

---

## Tech Stack
- Python (Pandas, NumPy)
- Scikit-Learn (Machine Learning Model + Scaler)
- Streamlit (Web Dashboard)
- PyDeck (Map Visualization)
- Matplotlib (Plots)

---

## Notes
- Prediction values are **approximate** and depend on weather + location.
- Dashboard is intended for **education and research only**, not medical or regulatory decisions.

