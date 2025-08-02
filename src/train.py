"""
Model training pipeline for flight delay prediction
"""
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score, accuracy_score
)

from data_preprocessing import FlightDataPreprocessor, create_sample_data
from config import MODEL_CONFIG, MODELS_DIR, TARGET_METRICS


class ModelTrainer:
    """Handles training and evaluation of multiple models"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_models(self):
        """Initialize all models with configured parameters"""
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(**MODEL_CONFIG['xgboost'])
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(**MODEL_CONFIG['lightgbm'])
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
        
        print(f"Initialized {len(self.models)} models")
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train a single model and return performance metrics"""
        print(f"\nTraining {model_name}...")
        
        model = self.models[model_name]
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate probabilities for AUC
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_f1': f1_score(y_train, train_pred),
            'val_f1': f1_score(y_val, val_pred),
            'train_precision': precision_score(y_train, train_pred),
            'val_precision': precision_score(y_val, val_pred),
            'train_recall': recall_score(y_train, train_pred),
            'val_recall': recall_score(y_val, val_pred),
            'train_auc': roc_auc_score(y_train, train_prob),
            'val_auc': roc_auc_score(y_val, val_prob),
            'training_time': training_time
        }
        
        self.model_performance[model_name] = metrics
        
        print(f"{model_name} Results:")
        print(f"  Validation F1: {metrics['val_f1']:.4f}")
        print(f"  Validation AUC: {metrics['val_auc']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        
        return metrics
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Perform cross-validation for all models"""
        print("\nPerforming cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")
            
            # F1 score cross-validation
            f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
            
            cv_results[model_name] = {
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'f1_scores': f1_scores
            }
            
            print(f"  {model_name} CV F1: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")
        
        return cv_results
    
    def train_all_models(self, X, y, test_size=0.2):
        """Train all models and compare performance"""
        print("Starting model training pipeline...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Class distribution - Train: {np.bincount(y_train)}")
        print(f"Class distribution - Val: {np.bincount(y_val)}")
        
        # Prepare models
        self.prepare_models()
        
        # Train each model
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train, X_val, y_val)
        
        # Perform cross-validation
        cv_results = self.cross_validate_models(X, y)
        
        # Select best model based on validation F1 score
        best_model_name = max(
            self.model_performance.keys(), 
            key=lambda x: self.model_performance[x]['val_f1']
        )
        
        self.best_model = self.models[best_model_name]
        self.best_score = self.model_performance[best_model_name]['val_f1']
        
        print(f"\nBest model: {best_model_name} (F1: {self.best_score:.4f})")
        
        # Check if target performance is met
        if self.best_score >= TARGET_METRICS['f1_score']:
            print(f"✅ Target F1 score of {TARGET_METRICS['f1_score']} achieved!")
        else:
            print(f"⚠️  Target F1 score of {TARGET_METRICS['f1_score']} not achieved")
        
        return self.models, self.model_performance, cv_results
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for the specified model"""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_models(self):
        """Save all trained models to disk"""
        MODELS_DIR.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = MODELS_DIR / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {model_path}")
        
        # Save performance metrics
        metrics_path = MODELS_DIR / "model_performance.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.model_performance, f)
        print(f"Saved performance metrics to {metrics_path}")
    
    def load_model(self, model_name):
        """Load a specific model from disk"""
        model_path = MODELS_DIR / f"{model_name}_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def generate_training_report(self, feature_names):
        """Generate a comprehensive training report"""
        print("\n" + "="*50)
        print("FLIGHT DELAY PREDICTION - TRAINING REPORT")
        print("="*50)
        
        print(f"\nTraining Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of models trained: {len(self.models)}")
        
        print(f"\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        for model_name, metrics in self.model_performance.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"  Validation F1 Score: {metrics['val_f1']:.4f}")
            print(f"  Validation Precision: {metrics['val_precision']:.4f}")
            print(f"  Validation Recall: {metrics['val_recall']:.4f}")
            print(f"  Validation AUC: {metrics['val_auc']:.4f}")
            print(f"  Training Time: {metrics['training_time']:.2f}s")
        
        # Feature importance for best model
        best_model_name = max(
            self.model_performance.keys(), 
            key=lambda x: self.model_performance[x]['val_f1']
        )
        
        print(f"\nTOP 10 FEATURES ({best_model_name.upper()}):")
        print("-" * 40)
        
        feature_importance = self.get_feature_importance(best_model_name, feature_names)
        if feature_importance is not None:
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nTARGET METRICS:")
        print("-" * 40)
        print(f"  Target F1 Score: {TARGET_METRICS['f1_score']}")
        print(f"  Best Achieved F1: {self.best_score:.4f}")
        print(f"  Status: {'✅ ACHIEVED' if self.best_score >= TARGET_METRICS['f1_score'] else '❌ NOT ACHIEVED'}")


def main():
    """Main training pipeline"""
    print("Flight Delay Prediction - Model Training")
    print("="*40)
    
    # Create sample data (in real implementation, load from files)
    print("Loading data...")
    flight_df, weather_df = create_sample_data()
    
    # Initialize preprocessor and process data
    print("Preprocessing data...")
    preprocessor = FlightDataPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(flight_df, weather_df)
    
    # Save preprocessor
    preprocessor.save_preprocessor(MODELS_DIR / "preprocessing.pkl")
    
    # Initialize trainer and train models
    trainer = ModelTrainer()
    models, performance, cv_results = trainer.train_all_models(X, y)
    
    # Save models
    trainer.save_models()
    
    # Generate report
    trainer.generate_training_report(feature_names)
    
    print(f"\nTraining completed! Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()