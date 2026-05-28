"""Simple model trainer for MVP."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class SimpleModelTrainer:
    """Random Forest model trainer with balanced class weights."""
    
    def __init__(self, model_path: str = './models'):
        """Initialize trainer."""
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.metrics = {}
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data: encode categoricals and scale numericals."""
        df = df.copy()
        
        # Separate features and target
        y = df['churn'].values
        X = df.drop(['customer_id', 'churn'], axis=1)
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numerical features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.feature_names = X.columns.tolist()
        return X_scaled, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train Random Forest model with balanced class weights."""
        print("🔄 Preprocessing data...")
        X, y = self.preprocess_data(df)
        
        print("📊 Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("🤖 Training Random Forest with balanced class weights...")
        self.model = self._build_model(y_train)
        self.model.fit(X_train, y_train)
        
        print("📈 Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = self._compute_metrics(y_test, y_pred, y_pred_proba)
        self._print_metrics()
        
        return self.metrics
    
    def _build_model(self, y_train: np.ndarray) -> RandomForestClassifier:
        """Build Random Forest with balanced class weights."""
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,
        )
    
    def _compute_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
    
    def _print_metrics(self) -> None:
        """Print model metrics."""
        print("\n✅ Model trained successfully!")
        for metric, value in self.metrics.items():
            print(f"   {metric.capitalize():10s}: {value:.3f}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on preprocessed features."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw features to model-ready array."""
        if not all([self.scaler, self.label_encoders, self.feature_names]):
            raise RuntimeError("Model preprocessing artifacts not loaded")

        df = df.copy()

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col not in df.columns:
                continue
            
            df[col] = df[col].astype(str)
            unseen_mask = ~df[col].isin(encoder.classes_)
            if unseen_mask.any():
                df.loc[unseen_mask, col] = encoder.classes_[0]
            df[col] = encoder.transform(df[col])

        # Ensure feature order matches training
        return self.scaler.transform(df[self.feature_names])
    
    def save(self, filename: str = 'model.pkl') -> None:
        """Save model and preprocessing artifacts."""
        filepath = self.model_path / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
            }, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filename: str = 'model.pkl') -> None:
        """Load model and preprocessing artifacts."""
        filepath = self.model_path / filename
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        
        print(f"✓ Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance sorted by value."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

