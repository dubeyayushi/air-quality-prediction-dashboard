"""
Streamlit App: PM2.5 Predictor Dashboard

Usage:
    streamlit run app.py

Dependencies:
    pip install streamlit pandas numpy scikit-learn joblib pydeck matplotlib

Default model path used in this project:
    /mnt/data/air_quality_model.pkl
"""

import io
import joblib
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# -------------------------
# Configuration
# -------------------------
DEFAULT_MODEL_PATH = "air_quality_model.pkl"  
EXPECTED_FEATURES = ["latitude", "longitude", "temperature", "humidity", "wind_speed"]
SAMPLE_ROW = {"latitude": 28.61, "longitude": 77.20, "temperature": 30.0, "humidity": 50.0, "wind_speed": 3.0}

# -------------------------
# Utilities
# -------------------------
@st.cache_data(show_spinner=False)
def load_model(path):
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

def predict_dataframe(model, df):
    # Ensure required feature columns exist (order matters)
    X = df[EXPECTED_FEATURES].copy()
    preds = model.predict(X.values)
    return preds

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def display_map(df, lat_col="latitude", lon_col="longitude", value_col="prediction"):
    # Build pydeck layer
    midpoint = (df[lat_col].mean(), df[lon_col].mean())
    layer = pdk.Layer(
        "ScatterplotLayer",
        df[[lon_col, lat_col, value_col]],
        get_position=[lon_col, lat_col],
        get_radius=20000,  # meters (scaled)
        radius_scale=10,
        get_fill_color="[255 - 2*prediction, 50, 2*prediction]",
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {"html": "<b>PM2.5:</b> {prediction}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}", "style": {"color": "white"}}
    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=6, pitch=0)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(r)


# -------------------------
# App layout
# -------------------------
st.set_page_config(page_title="PM2.5 Predictor — Smart City", layout="wide", initial_sidebar_state="expanded")
st.title("PM2.5 Predictor — Smart City Dashboard")
st.markdown("Predict PM2.5 using geolocation + weather features. Upload model or use default model path.")

# Sidebar: Model loading and options
st.sidebar.header("Model & Settings")
uploaded_model = st.sidebar.file_uploader("Upload model (`.pkl`) (optional)", type=["pkl"])
use_default = False
model = None
load_error = None

if uploaded_model is not None:
    # load model from uploaded file
    try:
        model = joblib.load(uploaded_model)
        st.sidebar.success("Model loaded from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")
else:
    st.sidebar.write(f"Default model path:\n`{DEFAULT_MODEL_PATH}`")
    if st.sidebar.button("Load default model"):
        model, load_error = load_model(DEFAULT_MODEL_PATH)
        if model:
            st.sidebar.success("Default model loaded.")
        else:
            st.sidebar.error(f"Failed to load default model: {load_error}")

st.sidebar.markdown("---")
st.sidebar.info("Expected features: " + ", ".join(EXPECTED_FEATURES))

# ---- Main: Tabs for Single / Batch
tab1, tab2, tab3 = st.tabs(["Single prediction", "Batch CSV prediction", "Model & Metrics"])

# -------------------------
# Tab 1: Single prediction
# -------------------------
with tab1:
    st.subheader("Single-point prediction")
    st.markdown("Enter the inputs manually (or use sample).")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=float(SAMPLE_ROW["latitude"]))
        lon = st.number_input("Longitude", value=float(SAMPLE_ROW["longitude"]))
        temp = st.number_input("Temperature (°C)", value=float(SAMPLE_ROW["temperature"]))
    with col2:
        humidity = st.number_input("Humidity (%)", value=float(SAMPLE_ROW["humidity"]))
        wind_speed = st.number_input("Wind speed (m/s)", value=float(SAMPLE_ROW["wind_speed"]))
        dummy = st.empty()

    # Load scaler once (after loading model)
    scaler = None
    if uploaded_model is not None:
        # model is already loaded
        scaler = joblib.load("scaler.pkl")  # uploaded separately (or ask user to upload)
    elif st.sidebar.button("Load default scaler"):
        scaler = joblib.load("/mnt/data/scaler.pkl")

    ...
    # Inside Single prediction block:

    if st.button("Predict single point"):
        if model is None or scaler is None:
            st.error("Load both model and scaler first.")
        else:
            # ---- engineered features (unchanged)
            from datetime import datetime
            now = datetime.utcnow()
            hour = now.hour
            day_of_week = now.weekday()
            month = now.month

            city_center_lat, city_center_lon = 28.6139, 77.2090
            distance_from_center = np.sqrt(
                (lat - city_center_lat) ** 2 + (lon - city_center_lon) ** 2
            )
            industrial_zone = int(
                ((lat < 28.65) and (lon > 77.25)) or
                ((lat < 28.55) and (lon < 77.20))
            )

            # 10 feature row in correct order
            input_row = np.array([[
                lat, lon, temp, humidity, wind_speed,
                distance_from_center, industrial_zone,
                hour, day_of_week, month
            ]])

            # ---- scale features exactly like training
            input_scaled = scaler.transform(input_row)

            # ---- final prediction
            pred = model.predict(input_scaled)[0]
            st.metric("Predicted PM2.5 (µg/m³)", f"{pred:.2f}")

            # Show predicted point on the map
            res_df = pd.DataFrame([{
                "latitude": lat,
                "longitude": lon,
                "prediction": float(pred)
            }])
            st.caption("Map view (predicted point)")
            display_map(res_df)



# -------------------------
# Tab 2: Batch CSV prediction
# -------------------------
with tab2:
    st.subheader("Batch prediction (CSV)")
    st.markdown("Upload CSV with columns: `latitude,longitude,temperature,humidity,wind_speed`. Optional `pm25` for metrics.")
    uploaded_csv = st.file_uploader("Upload CSV file (optional)", type=["csv"])
    sample_download = st.download_button(
        "Download sample CSV",
        data=df_to_csv_bytes(pd.DataFrame([SAMPLE_ROW])),
        file_name="sample_input.csv",
        mime="text/csv"
    )

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            # Check for required columns
            missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if st.button("Run batch prediction"):
                    if model is None:
                        st.error("No model loaded. Upload a .pkl file in the sidebar or load the default model.")
                    else:
                        preds = predict_dataframe(model, df)
                        df_result = df.copy()
                        df_result["prediction"] = preds
                        st.success("Predictions added to dataframe.")
                        st.dataframe(df_result.head())

                        # If ground truth provided compute metrics
                        if "pm25" in df_result.columns:
                            mae, rmse = compute_metrics(df_result["pm25"].values, df_result["prediction"].values)
                            st.write(f"MAE: {mae:.3f} — RMSE: {rmse:.3f}")

                        # Map
                        st.caption("Map for batch predictions")
                        display_map(df_result)

                        # Download button
                        csv_bytes = df_to_csv_bytes(df_result)
                        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        st.info("Upload a CSV file to run batch predictions, or use the sample CSV.")

# -------------------------
# Tab 3: Model & Metrics
# -------------------------
with tab3:
    st.subheader("Model details & quick validation")

    if model is None:
        st.warning("No model loaded. Upload a model in the sidebar or load default.")
    else:
        st.write("Model type:", type(model).__name__)
        # Try to show model attributes if possible
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fi_df = pd.DataFrame({"feature": EXPECTED_FEATURES, "importance": importances}).sort_values("importance", ascending=False)
                st.write("Feature importances:")
                st.table(fi_df)
                # Simple bar plot
                fig, ax = plt.subplots()
                ax.bar(fi_df["feature"], fi_df["importance"])
                ax.set_ylabel("Importance")
                ax.set_xlabel("Feature")
                st.pyplot(fig)
            else:
                st.info("Model does not expose `feature_importances_`. If it is a linear model you may inspect coefficients.")
                if hasattr(model, "coef_"):
                    coef = model.coef_.ravel() if hasattr(model.coef_, "ravel") else model.coef_
                    c_df = pd.DataFrame({"feature": EXPECTED_FEATURES, "coef": coef})
                    st.table(c_df)
        except Exception:
            st.info("Could not extract model internals (not necessary for prediction).")

    # Quick in-app validation using manual test data upload
    st.markdown("---")
    st.markdown("**Optional validation:** Upload CSV with `latitude,longitude,temperature,humidity,wind_speed,pm25` to compute metrics.")
    val_csv = st.file_uploader("Upload validation CSV (optional)", type=["csv"], key="val_csv")
    if val_csv is not None:
        try:
            val_df = pd.read_csv(val_csv)
            missing = [c for c in EXPECTED_FEATURES + ["pm25"] if c not in val_df.columns]
            if missing:
                st.error(f"Missing columns for validation: {missing}")
            else:
                if model is None:
                    st.error("Load model first.")
                else:
                    y_pred = predict_dataframe(model, val_df)
                    mae, rmse = compute_metrics(val_df["pm25"].values, y_pred)
                    st.metric("MAE", f"{mae:.3f}")
                    st.metric("RMSE", f"{rmse:.3f}")
                    # Show scatter
                    fig, ax = plt.subplots()
                    ax.scatter(val_df["pm25"], y_pred, alpha=0.6, s=8)
                    ax.plot([val_df["pm25"].min(), val_df["pm25"].max()], [val_df["pm25"].min(), val_df["pm25"].max()], 'r--')
                    ax.set_xlabel("True PM2.5")
                    ax.set_ylabel("Predicted PM2.5")
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Validation failed: {e}")

# -------------------------
# Footer / Notes
# -------------------------
st.markdown("---")
st.caption("Notes: The model predicts PM2.5 concentrations from geolocation and weather features. Use predictions for awareness and prototyping; do not use for medical advice or authoritative regulatory decisions.")
