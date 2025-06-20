import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_fraud_model():
    """Train and evaluate fraud detection model"""
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("🔄 Loading dataset...")
    try:
        df = pd.read_csv("data/creditcard.csv")
    except FileNotFoundError:
        print("❌ Error: creditcard.csv not found in data/ directory")
        print("📥 Please download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        return
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    # Data preprocessing
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    # Scale the Amount feature (V1-V28 are already scaled)
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🧠 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Model evaluation
    auc_score = roc_auc_score(y_test, y_pred_proba)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    print("\n📈 Model Performance:")
    print(f"🎯 AUC Score: {auc_score:.4f}")
    print(f"🔄 Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=feature_imp, x='importance', y='feature')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Model training completed!")
    print("📁 Files saved:")
    print("   - models/fraud_model.pkl")
    print("   - models/scaler.pkl")
    print("   - models/feature_names.pkl")
    print("   - reports/feature_importance.png")
    print("   - reports/confusion_matrix.png")
    
    return {
        'auc_score': auc_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

if __name__ == "__main__":
    train_fraud_model()