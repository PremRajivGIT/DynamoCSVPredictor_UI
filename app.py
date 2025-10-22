import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Global variables
current_model = None
current_scaler = None
current_imputer = None
current_features = []
current_target = ""
feature_ranges = {}
uploaded_df = None
original_selected_features = []  # Store original feature names before encoding

def load_california_dataset():
    global current_model, current_scaler, current_features, current_target, feature_ranges
    
    try:
        # Load California housing data
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['Price'] = housing.target * 100000 * 83  # Convert to INR
        
        X = df[housing.feature_names].values
        y = df['Price'].values
        
        current_features = list(housing.feature_names)
        current_target = 'Price (INR)'
        
        # Set feature ranges based on actual data
        feature_ranges = {
            'MedInc': (0.5, 15, 3.5, 0.1),
            'HouseAge': (1, 52, 25, 1),
            'AveRooms': (1, 10, 5, 0.5),
            'AveBedrms': (1, 5, 1, 0.5),
            'Population': (100, 5000, 1400, 100),
            'AveOccup': (1, 10, 3, 0.5),
            'Latitude': (32, 42, 37, 0.1),
            'Longitude': (-124, -114, -119, 0.1)
        }
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        current_model = model
        current_scaler = scaler
        
        score = r2_score(y_test, model.predict(X_test))
        
        # Save model
        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump({'feature_names': current_features, 'target_name': current_target}, 'metadata.pkl')
        
        # Return sliders for each feature
        sliders = []
        for feature in current_features:
            min_val, max_val, default_val, step_val = feature_ranges[feature]
            sliders.append(gr.Slider(
                minimum=min_val,
                maximum=max_val,
                value=default_val,
                step=step_val,
                label=feature,
                visible=True
            ))
        
        # Fill remaining slots with hidden sliders
        while len(sliders) < 10:
            sliders.append(gr.Slider(visible=False))
        
        return (f"✅ California Housing Dataset Loaded!\nR² Score: {score:.4f}\nSamples: {len(X)}\nFeatures: {len(current_features)}", 
                gr.update(visible=True), 
                gr.update(visible=False),  # Hide feature selection
                *sliders)
        
    except Exception as e:
        hidden_sliders = [gr.Slider(visible=False) for _ in range(10)]
        return f"❌ Error: {str(e)}", gr.update(visible=False), gr.update(visible=False), *hidden_sliders

def preview_csv(csv_file):
    global uploaded_df
    
    if csv_file is None:
        return "⚠️ Please upload a CSV file", gr.update(choices=[], value=[]), gr.update(choices=[], value=None), gr.update(visible=False)
    
    try:
        df = pd.read_csv(csv_file.name)
        uploaded_df = df
        
        # Get ALL columns
        all_cols = df.columns.tolist()
        
        if len(all_cols) < 2:
            return "❌ Error: CSV needs at least 2 columns", gr.update(choices=[], value=[]), gr.update(choices=[], value=None), gr.update(visible=False)
        
        # Build better preview
        preview_text = f"📊 CSV Preview:\n"
        preview_text += f"Total Rows: {len(df)}\n"
        preview_text += f"Total Columns: {len(all_cols)}\n\n"
        preview_text += f"Columns: {', '.join(all_cols)}\n\n"
        preview_text += "First 5 rows:\n"
        preview_text += df.head(5).to_string(index=True, max_cols=None, max_colwidth=20)
        preview_text += f"\n\n✅ All columns available for selection"
        
        # Default: all columns except last as features, last as target
        default_features = all_cols[:-1]
        default_target = all_cols[-1]
        
        return (preview_text, 
                gr.update(choices=all_cols, value=default_features),
                gr.update(choices=all_cols, value=default_target),
                gr.update(visible=True))
        
    except Exception as e:
        return f"❌ Error reading CSV: {str(e)}", gr.update(choices=[], value=[]), gr.update(choices=[], value=None), gr.update(visible=False)

def train_custom_model(selected_features, selected_target):
    global current_model, current_scaler, current_features, current_target, feature_ranges, uploaded_df
    
    if uploaded_df is None:
        hidden_sliders = [gr.Slider(visible=False) for _ in range(10)]
        return "❌ No CSV uploaded", gr.update(visible=False), *hidden_sliders
    
    if not selected_features or not selected_target:
        hidden_sliders = [gr.Slider(visible=False) for _ in range(10)]
        return "❌ Please select features and target column", gr.update(visible=False), *hidden_sliders
    
    try:
        # Extract selected columns
        X_df = uploaded_df[selected_features].copy()
        y_series = uploaded_df[selected_target]
        
        # Separate numeric and categorical columns
        numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Handle missing values in numeric columns first
        if numeric_features:
            imputer = SimpleImputer(strategy='mean')
            X_df[numeric_features] = imputer.fit_transform(X_df[numeric_features])
        
        # One-hot encode categorical columns
        if categorical_features:
            X_df = pd.get_dummies(X_df, columns=categorical_features, drop_first=True)
        
        X = X_df.values
        y = y_series.values
        
        # Handle NaN in target
        y_mask = pd.notna(y)
        if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number):
            y_mask = y_mask & ~np.isinf(y)
            
        X = X[y_mask]
        y = y[y_mask]
        
        if len(X) < 10:
            hidden_sliders = [gr.Slider(visible=False) for _ in range(10)]
            return "❌ Not enough valid data after cleaning", gr.update(visible=False), *hidden_sliders
        
        current_features = list(X_df.columns)
        current_target = selected_target
        
        # Auto-detect feature ranges (for encoded features)
        feature_ranges = {}
        for i, feature in enumerate(current_features):
            col_min = X[:, i].min()
            col_max = X[:, i].max()
            col_median = np.median(X[:, i])
            
            feature_ranges[feature] = (
                float(col_min),
                float(col_max),
                float(col_median),
                max((col_max - col_min) / 100, 0.1)
            )
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        current_model = model
        current_scaler = scaler
        
        score = r2_score(y_test, model.predict(X_test))
        
        # Save model
        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump({'feature_names': current_features, 'target_name': current_target}, 'metadata.pkl')
        
        # Return sliders for each feature
        sliders = []
        for feature in current_features:
            min_val, max_val, default_val, step_val = feature_ranges[feature]
            sliders.append(gr.Slider(
                minimum=min_val,
                maximum=max_val,
                value=default_val,
                step=step_val,
                label=feature,
                visible=True
            ))
        
        # Fill remaining slots with hidden sliders
        while len(sliders) < 10:
            sliders.append(gr.Slider(visible=False))
        
        return (f"✅ Custom Model Trained!\nR² Score: {score:.4f}\nSamples: {len(X)}\nFeatures: {', '.join(current_features)}\nTarget: {current_target}", 
                gr.update(visible=True), *sliders)
        
    except Exception as e:
        hidden_sliders = [gr.Slider(visible=False) for _ in range(10)]
        return f"❌ Error: {str(e)}", gr.update(visible=False), *hidden_sliders

def predict_price(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):
    if current_model is None:
        return "❌ No model loaded"
    
    try:
        # Collect only the visible inputs based on current features
        inputs = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10][:len(current_features)]
        
        features = np.array([inputs])
        features_scaled = current_scaler.transform(features)
        prediction = current_model.predict(features_scaled)[0]
        return f"₹{prediction:,.2f}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Noir Gold Theme (fixed: removed invalid variables, added body_text_color for white text visibility on dark bg)
noir_gold = gr.themes.Monochrome(
    primary_hue="yellow",
    secondary_hue="amber",
    neutral_hue="gray",
).set(
    body_background_fill="#0A0A0A",
    body_background_fill_dark="#0A0A0A",
    body_text_color="#FFFFFF",
    body_text_color_dark="#FFFFFF",
    block_background_fill="#141414",
    block_background_fill_dark="#141414",
    block_border_color="#333333",
    block_border_color_dark="#333333",
    block_label_text_color="#FFD700",
    block_label_text_color_dark="#FFD700",
    button_primary_background_fill="#FFD700",
    button_primary_background_fill_hover="#E6C200",
    button_primary_background_fill_dark="#FFD700",
    button_primary_text_color="#0A0A0A",
    button_primary_text_color_dark="#0A0A0A",
    button_secondary_background_fill="#1A1A1A",
    button_secondary_background_fill_hover="#262626",
    button_secondary_text_color="#FFD700",
    input_background_fill="#1A1A1A",
    input_background_fill_dark="#1A1A1A",
    input_border_color="#333333",
    checkbox_label_text_color="#FFD700",
    slider_color="#FFD700",
    shadow_drop_lg="0 0 10px rgba(255, 215, 0, 0.2)",
)

# Main Gradio Interface
with gr.Blocks(theme=noir_gold, title="Real Estate Price Predictor") as demo:
    gr.Markdown("# 🏠 Real Estate Price Predictor")
    gr.Markdown("### Select your dataset to begin")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 📊 California Housing Dataset")
            gr.Markdown("Built-in dataset with 8 features")
            california_btn = gr.Button("Load California Dataset", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("#### 📁 Custom Dataset")
            gr.Markdown("Upload your own CSV file")
            csv_upload = gr.File(label="Upload CSV", file_types=[".csv"])
            preview_btn = gr.Button("Preview CSV", variant="secondary", size="lg")
    
    status_output = gr.Textbox(label="Status", lines=6, interactive=False)
    
    # Feature Selection Section (only for custom CSV)
    with gr.Column(visible=False) as feature_selection:
        gr.Markdown("---")
        gr.Markdown("## 🎯 Select Features & Target")
        
        feature_selector = gr.CheckboxGroup(label="Select Features (input columns)", choices=[])
        target_selector = gr.Dropdown(label="Select Target (output column)", choices=[])
        train_custom_btn = gr.Button("Train Model", variant="primary", size="lg")
    
    # Prediction Section (hidden until dataset is loaded)
    with gr.Column(visible=False) as prediction_section:
        gr.Markdown("---")
        gr.Markdown("## 💰 Make Predictions")
        
        # Create 10 slider slots (max features supported)
        slider1 = gr.Slider(visible=False)
        slider2 = gr.Slider(visible=False)
        slider3 = gr.Slider(visible=False)
        slider4 = gr.Slider(visible=False)
        slider5 = gr.Slider(visible=False)
        slider6 = gr.Slider(visible=False)
        slider7 = gr.Slider(visible=False)
        slider8 = gr.Slider(visible=False)
        slider9 = gr.Slider(visible=False)
        slider10 = gr.Slider(visible=False)
        
        predict_btn = gr.Button("Predict Price", variant="primary", size="lg")
        prediction_output = gr.Textbox(label="Predicted Price", lines=2, interactive=False)
    
    # Button actions
    california_btn.click(
        fn=load_california_dataset,
        inputs=[],
        outputs=[status_output, prediction_section, feature_selection, slider1, slider2, slider3, slider4, slider5, slider6, slider7, slider8, slider9, slider10]
    )
    
    preview_btn.click(
        fn=preview_csv,
        inputs=[csv_upload],
        outputs=[status_output, feature_selector, target_selector, feature_selection]
    )
    
    train_custom_btn.click(
        fn=train_custom_model,
        inputs=[feature_selector, target_selector],
        outputs=[status_output, prediction_section, slider1, slider2, slider3, slider4, slider5, slider6, slider7, slider8, slider9, slider10]
    )
    
    # Prediction logic
    predict_btn.click(
        fn=predict_price,
        inputs=[slider1, slider2, slider3, slider4, slider5, slider6, slider7, slider8, slider9, slider10],
        outputs=prediction_output
    )

if __name__ == "__main__":
    demo.launch()
