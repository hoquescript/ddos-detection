import pandas as pd
import numpy as np
import time
import os
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Go up 3 levels: model -> src -> ddos-detection (Project Root)
ROOT = Path(__file__).parent

# Set to True when running on Compute Canada with full dataset
USE_FULL_DATA = False

def load_data(project_root=ROOT):
    if USE_FULL_DATA:
        # Source directory for raw CSVs
        dataset_source = project_root / 'data'
        input_files = list(dataset_source.glob("*.csv"))

        if not input_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_source}")

        # Merge all datasets
        merged_datasets = []
        for file in input_files:
            print(f"Loading {file.name}...")
            df = pd.read_csv(file, low_memory=False)
            merged_datasets.append(df)
        return pd.concat(merged_datasets, ignore_index=True)
    else:
        # Load the pre-generated sample file
        sample_file = project_root / 'data' / 'processed' / 'sample_dataset.csv'

        if not sample_file.exists():
            raise FileNotFoundError(f"Sample file not found at {sample_file}")
        
        return pd.read_csv(sample_file, low_memory=False)


def run_experiment():
    # Create output directories if they don't exist
    output_dir = ROOT / 'output'
    models_dir = ROOT / 'models'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    #### Load Data ####
    print("Loading data...")
    df = load_data()

    #### Cleanup Dataset ####
    print("Cleaning data...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if df.isna().sum().sum() > 0:
        df.dropna(inplace=True)
    
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    #### Feature Selection ####
    # Encoding Target Labels
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    # Store class names for reporting
    class_names = [str(cls) for cls in le.classes_]
    print(f"Classes found: {class_names}")

    # Prepare Features
    X = df.drop(columns=['label'])

    # Removing non-numeric features
    non_numeric_columns = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_columns) > 0:
        X = X.drop(columns=non_numeric_columns)

    # Standardization
    print("Standardizing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Principal Component Analysis
    print("Running PCA...")
    pca = PCA(n_components=24)
    X_pca = pca.fit_transform(X_scaled)

    # --- CRITICAL ADDITION FOR DEMO ---
    print("Saving preprocessors for demo...")
    joblib.dump(le, models_dir / 'label_encoder.joblib')
    joblib.dump(scaler, models_dir / 'scaler.joblib')
    joblib.dump(pca, models_dir / 'pca.joblib')
    # ----------------------------------

    #### Train model ####
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=42)

    # Model Selection
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Random_Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True), 
        "Naive_Bayes": GaussianNB(),
        "Decision_Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    print("Starting Training Loop...")
    for name, model in models.items():
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"Training Complete: {name}")

            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate AUC safely
            try:
                y_proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', labels=range(len(class_names)))
            except:
                auc = 0.0 

            # Get metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0)
            # Saving the metrics
            file_path = os.path.join(output_dir, f'{name}.txt')
            with open(file_path, 'w') as f:
                f.write(f"--- METRICS FOR {name.upper()} ---\n")
                f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
                f.write(f"AUC: {auc:.4f}\n")
                f.write(f"Execution Time: {execution_time:.4f}s\n\n")
                f.write(report)

            print(f"Successfully saved metrics to {file_path}")

            # Save the Trained Model
            model_path = os.path.join(models_dir, f'{name}.joblib')
            joblib.dump(model, model_path)
            
        except Exception as e:
            print(f"  -> ERROR training {name}: {e}")

if __name__ == "__main__":
    run_experiment()