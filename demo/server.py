from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path

app = Flask(__name__)

# --- CONFIGURATION ---
# Adjust this path if your folder structure is different
# Currently pointing to: ddos-detection/data/processed/sample_dataset.csv
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
DATA_FILE = ROOT / 'data' / 'processed' / 'sample_dataset.csv'

# Global variables
scaler = None
pca = None
le = None
model = None
traffic_data = None

def load_resources():
    global scaler, pca, le, model, traffic_data
    print("Loading Defense System...")
    
    try:
        # Load Preprocessors and Model
        scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
        pca = joblib.load(MODELS_DIR / 'pca.joblib')
        le = joblib.load(MODELS_DIR / 'label_encoder.joblib')
        model = joblib.load(MODELS_DIR / 'Random_Forest.joblib')
        
        # Load Traffic Data
        traffic_data = pd.read_csv(DATA_FILE, low_memory=False)
        
        # --- CRITICAL FIX: CLEAN COLUMN NAMES ---
        # 1. Strip spaces (" Label" -> "Label")
        # 2. Lowercase ("Label" -> "label")
        # 3. Replace spaces with underscores ("Flow ID" -> "flow_id")
        traffic_data.columns = traffic_data.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Clean data content
        traffic_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        traffic_data.dropna(inplace=True)
        
        print(f"✅ System Loaded. Columns found: {list(traffic_data.columns)}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading resources: {e}")
        print("Ensure you ran ddos_experiment.py first!")

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/get_traffic')
def get_traffic():
    global scaler, pca, le, model, traffic_data
    
    if model is None or traffic_data is None:
        return jsonify({'error': 'System not initialized'}), 500

    try:
        # 1. Pick a random packet
        packet_row = traffic_data.sample(1)
        
        # 2. Extract features (Safely drop label)
        # Now that columns are cleaned, 'label' should exist
        if 'label' in packet_row.columns:
            features_raw = packet_row.drop(columns=['label'])
        else:
            # Fallback: If label is missing, just use all numeric cols
            features_raw = packet_row
            
        features_numeric = features_raw.select_dtypes(include=['number'])
        
        # 3. Start Timer
        start_time = time.perf_counter()
        
        # 4. Predict
        features_scaled = scaler.transform(features_numeric)
        features_pca = pca.transform(features_scaled)
        prediction_idx = model.predict(features_pca)[0]
        prediction_label = le.inverse_transform([prediction_idx])[0]
        
        confidence = np.max(model.predict_proba(features_pca))
        
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 5. Generate Fake IP for demo realism
        src_ip = f"192.168.1.{random.randint(2, 254)}"
        protocol = "TCP" if random.random() > 0.5 else "UDP"
        
        return jsonify({
            'src_ip': src_ip,
            'protocol': protocol,
            'prediction': prediction_label,
            'is_threat': bool(prediction_label != 'BENIGN'),
            'confidence': f"{confidence*100:.1f}%",
            'latency': f"{latency_ms:.4f} ms",
            'timestamp': time.strftime("%H:%M:%S")
        })

    except Exception as e:
        print(f"Error processing packet: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=3001)