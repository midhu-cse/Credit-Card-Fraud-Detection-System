import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_data():
    """Create sample fraud detection data if Kaggle dataset not available"""
    print("📊 Creating sample fraud detection dataset...")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Generate normal transactions (95%)
    n_normal = int(n_samples * 0.95)
    normal_data = np.random.normal(0, 1, (n_normal, 28))
    normal_amounts = np.random.lognormal(3, 1, n_normal)  # Typical amounts
    normal_labels = np.zeros(n_normal)
    
    # Generate fraudulent transactions (5%)
    n_fraud = n_samples - n_normal
    fraud_data = np.random.normal(0, 2, (n_fraud, 28))  # More extreme values
    fraud_amounts = np.random.lognormal(5, 1.5, n_fraud)  # Higher amounts
    fraud_labels = np.ones(n_fraud)
    
    # Combine data
    X = np.vstack([normal_data, fraud_data])
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    y = np.concatenate([normal_labels, fraud_labels])
    
    # Create DataFrame
    columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    df = pd.DataFrame(np.column_stack([X, amounts]), columns=columns)
    df['Class'] = y.astype(int)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save sample data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/creditcard.csv", index=False)
    print(f"✅ Sample dataset created: {len(df)} transactions")
    print(f"   - Normal: {(df['Class'] == 0).sum()}")
    print(f"   - Fraud: {(df['Class'] == 1).sum()}")
    
    return df

def train_model_standalone():
    """Train fraud detection model"""
    print("🤖 Starting Fraud Detection Model Training...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Try to load Kaggle dataset, create sample if not available
    try:
        print("📂 Loading Kaggle dataset...")
        df = pd.read_csv("data/creditcard.csv")
        print(f"✅ Loaded Kaggle dataset: {len(df)} transactions")
    except FileNotFoundError:
        print("⚠️  Kaggle dataset not found. Creating sample dataset...")
        df = create_sample_data()
    
    print(f"📊 Dataset Info:")
    print(f"   - Total transactions: {len(df)}")
    print(f"   - Normal transactions: {(df['Class'] == 0).sum()}")
    print(f"   - Fraudulent transactions: {(df['Class'] == 1).sum()}")
    print(f"   - Fraud rate: {df['Class'].mean()*100:.2f}%")
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in ['Class', 'Time']]
    X = df[feature_columns].copy()
    y = df['Class']
    
    # Scale Amount feature (V1-V28 should already be scaled in Kaggle data)
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🧠 Training Random Forest model...")
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',  # Handle imbalanced data
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print("\n📈 Model Performance:")
    print(f"🎯 AUC Score: {auc_score:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model components
    print("\n💾 Saving model components...")
    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    # Feature importance
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.subplot(2, 1, 1)
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    
    # Confusion matrix
    plt.subplot(2, 1, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('reports/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save model summary
    with open('reports/model_summary.txt', 'w') as f:
        f.write("Credit Card Fraud Detection Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Features: {len(X.columns)}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"AUC Score: {auc_score:.4f}\n\n")
        f.write("Feature Names:\n")
        for i, feature in enumerate(X.columns, 1):
            f.write(f"{i:2d}. {feature}\n")
    
    print("✅ Model training completed successfully!")
    print("\n📁 Generated files:")
    print("   - models/fraud_model.pkl")
    print("   - models/scaler.pkl") 
    print("   - models/feature_names.pkl")
    print("   - reports/model_performance.png")
    print("   - reports/model_summary.txt")
    
    print(f"\n🎯 Final Model Performance:")
    print(f"   - AUC Score: {auc_score:.4f}")
    print(f"   - Model ready for deployment!")
    
    return model, scaler, list(X.columns), auc_score

if __name__ == "__main__":
    # Run the model training
    model, scaler, features, auc = train_model_standalone()
    
    # Test the saved model
    print("\n🧪 Testing saved model...")
    try:
        loaded_model = joblib.load("models/fraud_model.pkl")
        loaded_scaler = joblib.load("models/scaler.pkl")
        loaded_features = joblib.load("models/feature_names.pkl")
        
        # Create a test sample
        test_sample = np.random.normal(0, 1, 29).reshape(1, -1)
        test_sample[0, -1] = 100.0  # Set amount
        
        # Scale amount
        test_df = pd.DataFrame(test_sample, columns=loaded_features)
        test_df['Amount'] = loaded_scaler.transform(test_df[['Amount']])
        
        # Make prediction
        prediction = loaded_model.predict(test_df)[0]
        probability = loaded_model.predict_proba(test_df)[0][1]
        
        print(f"✅ Test prediction successful!")
        print(f"   - Prediction: {'Fraud' if prediction else 'Normal'}")
        print(f"   - Probability: {probability:.4f}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")