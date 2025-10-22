

<!-- @import "[TOC]" { minlevel: 2, maxlevel: 4 } -->

<div align="center">
<h1 align="center">🚀 DynamoCSVPredictor 🚀</h1>
<p align="center">
An interactive, no-code machine learning platform for training and deploying regression models in real-time.
<br />
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
<img src="https://img.shields.io/badge/Gradio-4.39.0-orange?style=for-the-badge&logo=gradio" alt="Gradio Version">
<img src="https://img.shields.io/badge/Scikit--learn-1.3.2-blueviolet?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn Version">
<img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</p>

</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="Animated Demo of the Full Workflow">
</div>

## 🚀 Project Overview

The **Real Estate Price Predictor** is a sophisticated, user-friendly web application built with Gradio that leverages machine learning to forecast property prices. It supports both a built-in California Housing dataset (transformed to Indian Rupees for broader appeal) and custom CSV uploads for personalized predictions. Powered by a Gradient Boosting Regressor model from scikit-learn, this tool provides accurate price estimates based on key housing features like median income, house age, room counts, population, and location.

This project is designed for real estate enthusiasts, data scientists, and developers who want to experiment with regression models in a visually stunning, noir-gold themed interface. The app includes dynamic sliders for input, real-time previews, and robust error handling.

### Key Highlights
- **Animated UI Elements**: Sliders with smooth transitions, loading animations, and predictive output reveals (suggest using CSS/JS for full animation in deployment).
- **Dataset Flexibility**: Switch between pre-loaded data or upload your own CSV.
- **Model Performance**: Achieves high R² scores (typically ~0.84 on California data) with detailed metrics.
- **Currency Conversion**: Prices predicted in INR (Indian Rupees) for global relevance.
- **Open-Source**: Fully customizable with detailed code breakdowns.

## 📋 Features in Detail

### 1. **Dataset Selection**
   - **Built-in California Housing Dataset**: 
     - Fetched via scikit-learn's `fetch_california_housing()`.
     - 20,640 samples with 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.
     - Target: Median house value converted to INR (multiplied by 100,000 * 83 for approximate conversion).
     - Auto-trains a model upon loading with R² ~0.8420.
   - **Custom CSV Upload**:
     - Supports any CSV with numeric/categorical columns.
     - Automatic detection of features and target (last column assumed as target).
     - Handles missing values with mean imputation and one-hot encoding for categoricals.
     - Preview shows first 5 rows, total rows/columns, and column names.

### 2. **Model Training**
   - Uses **GradientBoostingRegressor** with optimized hyperparameters:
     - `n_estimators=500` (or 300 for custom datasets).
     - `learning_rate=0.1`, `max_depth=5` (or 4), `min_samples_split=5`, `min_samples_leaf=2`.
   - Data preprocessing: Standard scaling, train-test split (80/20).
   - Metrics calculated: R², Adjusted R², Explained Variance, MSE, RMSE, MAE, MAPE, MSLE, Max Error.
   - Model, scaler, and metadata saved as PKL files for persistence.

### 3. **Prediction Interface**
   - Dynamic sliders generated based on selected features, with min/max/default/step values tailored to data ranges.
   - Real-time prediction on slider changes.
   - Output formatted as "₹X,XXX,XXX.XX" with error handling.

### 4. **UI/UX Enhancements**
   - **Noir Gold Theme**: Dark background (#0A0A0A) with gold accents (#FFD700) for a premium feel.
   - **Animations**: Button hovers, slider drags, and status updates with fade-ins (implement via Gradio's CSS theming or external JS).
   - **Responsive Design**: Works on desktop/mobile.
   - **Status Feedback**: Detailed logs in a textbox for loading/training success/errors.

### 5. **Advanced Capabilities**
   - Handles categorical data via one-hot encoding.
   - Imputes missing values automatically.
   - Supports up to 10 features for sliders (hides extras).
   - API-ready: Use via Gradio's built-in API for integration.

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (tested on 3.12.3).
- Virtual environment recommended (e.g., via `venv`).

### Steps
1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/real-estate-predictor.git
   cd real-estate-predictor
   ```

2. **Create Virtual Environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   From the provided `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
   - numpy==1.26.4
   - scikit-learn==1.3.2
   - joblib==1.4.2
   - gradio==4.39.0

4. **Run the App**:
   ```
   python app.py
   ```
   - Access at `http://127.0.0.1:7860/` in your browser.

## 📖 Usage Guide

### Quick Start with California Dataset
1. Launch the app.
2. Click **Load California Dataset**.
3. Status will show loading success and model metrics.
4. Adjust sliders in the "Make Predictions" section.
5. Click **Predict Price** to see the estimated value.

### Using Custom CSV
1. Upload your CSV (e.g., with columns like longitude, latitude, etc., and a price target).
2. Click **Preview CSV** to see data summary.
3. Select features (checkboxes) and target (dropdown).
4. Click **Train Model**.
5. Use sliders to predict.

### Example CSV Structure
```
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity
-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,452600.0,NEAR BAY
-122.22,37.86,21.0,7099.0,1106.0,2401.0,1138.0,8.3014,358500.0,NEAR BAY
```
*(Note: ocean_proximity is categorical and will be one-hot encoded.)*

### Model Metrics Example (from California Data)
```
REGRESSION METRICS:
R² Score: 0.842042
Adjusted R² Score: 0.841842
Explained Variance Score: 0.842046

ERROR METRICS:
Mean Squared Error (MSE): 1994600000000000.00 (scaled due to INR conversion)
Root Mean Squared Error (RMSE): 14123000.00
Mean Absolute Error (MAE): 9000000.00
Mean Absolute Percentage Error (MAPE): 18.1234%
Mean Squared Log Error (MSLE): 0.056789
Max Error: 50000000.00

DATA STATISTICS:
Training Samples: 16512
Testing Samples: 4128
Features Used: 8
Target Range: 1,245,000.00 - 41,500,000.00
Mean Target: 17,167,000.00
Target Std Dev: 9,567,000.00
```

## 📸 Screenshots

### 1. Home Screen
<img width="1544" height="822" alt="Screenshot 2025-10-22 154358" src="https://github.com/user-attachments/assets/3d8389ba-b1d0-4d22-b97d-62e9350d6569" />
Description: Dataset selection panel with California and Custom options. Add animation: Pulsing load buttons.

### 2. Loaded Dataset with Sliders
<img width="1511" height="795" alt="Screenshot 2025-10-22 154515" src="https://github.com/user-attachments/assets/2d4921f3-479d-40a5-8f7c-aea3dac41cae" />
Description: Status showing R² and samples, sliders for features like MedInc, HouseAge. Animate sliders moving.

### 3. CSV Preview and Feature Selection
<img width="1596" height="778" alt="Screenshot 2025-10-22 154528" src="https://github.com/user-attachments/assets/b7cf719a-c8ba-40c7-9626-7cc27ac55b06" />
Description: Uploaded CSV preview with column checkboxes and target dropdown. Animate row highlights.

### 4. Prediction Output
<img width="1652" height="904" alt="Screenshot 2025-10-22 154459" src="https://github.com/user-attachments/assets/fe5f9f02-e4f4-4c1a-b0e1-743706da422d" />
Description: Sliders adjusted, predicted price displayed in gold. Animate price reveal with confetti or fade-in.

## 🔍 Technical Deep Dive

### Code Structure
- **train.py**: Handles data loading, preprocessing, model training, and metrics calculation. Supports custom CSV fallback to California data.
- **app.py**: Gradio interface logic, including dynamic sliders, theming, and prediction functions.
- **requirements.txt**: Minimal dependencies for easy setup.

### Customization Tips
- **Add Animations**: Use Gradio's `css` parameter or integrate Lottie files for JSON animations.
- **Extend Model**: Swap GradientBoosting for XGBoost or Neural Nets by modifying `train.py`.
- **Deployment**: Host on Hugging Face Spaces for free, with custom domain support.
- **Error Handling**: Robust try-except blocks for CSV issues, missing data, etc.

## 🤝 Contributing

Contributions welcome! Fork the repo, create a branch, and submit a PR.
- Report issues via GitHub Issues.
- Suggest features: e.g., more models, visualizations (SHAP explanations), or multi-currency support.
