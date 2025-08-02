"""
Model evaluation and validation framework for flight delay prediction
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_preprocessing import FlightDataPreprocessor, create_sample_data
from train import ModelTrainer
from predict import FlightDelayPredictor
from config import MODELS_DIR, TARGET_METRICS


class ModelEvaluator:
    """Comprehensive model evaluation and validation"""
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.test_scenarios = []
        
    def load_models(self):
        """Load all trained models"""
        model_types = ['xgboost', 'lightgbm', 'random_forest']
        
        for model_type in model_types:
            try:
                predictor = FlightDelayPredictor(model_name=model_type)
                predictor.load_model()
                self.models[model_type] = predictor
                print(f"Loaded {model_type} model")
            except FileNotFoundError:
                print(f"Warning: {model_type} model not found")
    
    def evaluate_model_performance(self, X_test, y_test, model_name):
        """Evaluate single model performance"""
        model = self.models[model_name].model
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'avg_precision': average_precision_score(y_test, y_prob)
        }
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics.update({
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        })
        
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_test
        }
        
        return metrics
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Perform cross-validation for all models"""
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, predictor in self.models.items():
            model = predictor.model
            
            # Cross-validation for multiple metrics
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            model_cv_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
                model_cv_results[f'{metric}_scores'] = scores
                model_cv_results[f'{metric}_mean'] = scores.mean()
                model_cv_results[f'{metric}_std'] = scores.std()
            
            cv_results[model_name] = model_cv_results
            
            print(f"{model_name} CV Results:")
            print(f"  F1: {model_cv_results['f1_mean']:.4f} (±{model_cv_results['f1_std']:.4f})")
            print(f"  AUC: {model_cv_results['roc_auc_mean']:.4f} (±{model_cv_results['roc_auc_std']:.4f})")
        
        return cv_results
    
    def generate_classification_report(self, model_name):
        """Generate detailed classification report"""
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        y_true = self.evaluation_results[model_name]['true_labels']
        y_pred = self.evaluation_results[model_name]['predictions']
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return report, cm
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """Plot confusion matrix"""
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        y_true = self.evaluation_results[model_name]['true_labels']
        y_pred = self.evaluation_results[model_name]['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Delay', 'Predicted Delay'],
            y=['Actual No Delay', 'Actual Delay'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name.upper()}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400,
            width=500
        )
        
        return fig
    
    def plot_roc_curve(self, model_names=None):
        """Plot ROC curves for specified models"""
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        fig = go.Figure()
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
                
            y_true = self.evaluation_results[model_name]['true_labels']
            y_prob = self.evaluation_results[model_name]['probabilities']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name.upper()} (AUC = {auc_score:.3f})',
                line=dict(width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            width=600
        )
        
        return fig
    
    def plot_precision_recall_curve(self, model_names=None):
        """Plot Precision-Recall curves"""
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        fig = go.Figure()
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
                
            y_true = self.evaluation_results[model_name]['true_labels']
            y_prob = self.evaluation_results[model_name]['probabilities']
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=f'{model_name.upper()} (AP = {avg_precision:.3f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            width=600
        )
        
        return fig
    
    def create_test_scenarios(self):
        """Create specific test scenarios for validation"""
        scenarios = [
            {
                'name': 'Snowstorm Scenario',
                'description': 'JFK to ORD in December with snow',
                'flight_data': {
                    'origin': 'JFK',
                    'destination': 'ORD',
                    'date': '2023-12-15',
                    'crs_dep_time': 800,
                    'weather_type': 'Snow',
                    'precipitation_in': 0.8
                },
                'expected_result': 'High',
                'expected_probability_range': (0.7, 1.0)
            },
            {
                'name': 'Clear Weather Scenario',
                'description': 'LAX to SFO in July with clear weather',
                'flight_data': {
                    'origin': 'LAX',
                    'destination': 'SFO',
                    'date': '2023-07-15',
                    'crs_dep_time': 1400,
                    'weather_type': 'Clear',
                    'precipitation_in': 0.0
                },
                'expected_result': 'Low',
                'expected_probability_range': (0.0, 0.2)
            },
            {
                'name': 'Thunderstorm Scenario',
                'description': 'ATL to MIA in summer with thunderstorms',
                'flight_data': {
                    'origin': 'ATL',
                    'destination': 'MIA',
                    'date': '2023-08-20',
                    'crs_dep_time': 1800,
                    'weather_type': 'Thunderstorm',
                    'precipitation_in': 0.6
                },
                'expected_result': 'High',
                'expected_probability_range': (0.6, 1.0)
            },
            {
                'name': 'Light Rain Scenario',
                'description': 'SEA to PDX with light rain',
                'flight_data': {
                    'origin': 'SEA',
                    'destination': 'PDX',
                    'date': '2023-11-10',
                    'crs_dep_time': 1000,
                    'weather_type': 'Rain',
                    'precipitation_in': 0.2
                },
                'expected_result': 'Medium',
                'expected_probability_range': (0.3, 0.7)
            }
        ]
        
        self.test_scenarios = scenarios
        return scenarios
    
    def validate_test_scenarios(self):
        """Validate models against test scenarios"""
        if not self.test_scenarios:
            self.create_test_scenarios()
        
        scenario_results = {}
        
        for model_name, predictor in self.models.items():
            model_results = []
            
            for scenario in self.test_scenarios:
                try:
                    result = predictor.predict_single_flight(scenario['flight_data'])
                    
                    # Check if prediction matches expected result
                    prediction_correct = result['prediction'] == scenario['expected_result']
                    
                    # Check if probability is in expected range
                    prob_min, prob_max = scenario['expected_probability_range']
                    probability_correct = prob_min <= result['probability'] <= prob_max
                    
                    model_results.append({
                        'scenario': scenario['name'],
                        'prediction': result['prediction'],
                        'probability': result['probability'],
                        'expected_prediction': scenario['expected_result'],
                        'expected_prob_range': scenario['expected_probability_range'],
                        'prediction_correct': prediction_correct,
                        'probability_correct': probability_correct,
                        'overall_correct': prediction_correct and probability_correct
                    })
                    
                except Exception as e:
                    model_results.append({
                        'scenario': scenario['name'],
                        'error': str(e),
                        'overall_correct': False
                    })
            
            scenario_results[model_name] = model_results
        
        return scenario_results
    
    def calculate_performance_score(self, model_name):
        """Calculate overall performance score"""
        if model_name not in self.evaluation_results:
            return 0.0
        
        metrics = self.evaluation_results[model_name]['metrics']
        
        # Weighted performance score
        weights = {
            'f1_score': 0.4,
            'roc_auc': 0.3,
            'precision': 0.15,
            'recall': 0.15
        }
        
        score = sum(metrics[metric] * weight for metric, weight in weights.items())
        return score
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("FLIGHT DELAY PREDICTION - EVALUATION REPORT")
        print("="*60)
        
        print(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models Evaluated: {len(self.evaluation_results)}")
        
        # Performance summary
        print(f"\nPERFORMANCE SUMMARY:")
        print("-" * 40)
        
        performance_scores = {}
        for model_name in self.evaluation_results:
            metrics = self.evaluation_results[model_name]['metrics']
            score = self.calculate_performance_score(model_name)
            performance_scores[model_name] = score
            
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Precision:    {metrics['precision']:.4f}")
            print(f"  Recall:       {metrics['recall']:.4f}")
            print(f"  F1 Score:     {metrics['f1_score']:.4f}")
            print(f"  ROC AUC:      {metrics['roc_auc']:.4f}")
            print(f"  Performance:  {score:.4f}")
            
            # Check target achievement
            target_met = "✅" if metrics['f1_score'] >= TARGET_METRICS['f1_score'] else "❌"
            print(f"  Target F1:    {target_met} ({TARGET_METRICS['f1_score']:.3f})")
        
        # Best model
        best_model = max(performance_scores.keys(), key=lambda x: performance_scores[x])
        print(f"\nBEST MODEL: {best_model.upper()} (Score: {performance_scores[best_model]:.4f})")
        
        # Scenario validation results
        scenario_results = self.validate_test_scenarios()
        print(f"\nSCENARIO VALIDATION:")
        print("-" * 40)
        
        for model_name, results in scenario_results.items():
            correct_predictions = sum(1 for r in results if r.get('overall_correct', False))
            total_scenarios = len(results)
            accuracy = correct_predictions / total_scenarios if total_scenarios > 0 else 0
            
            print(f"{model_name.upper()}: {correct_predictions}/{total_scenarios} scenarios correct ({accuracy:.1%})")
        
        return {
            'performance_scores': performance_scores,
            'best_model': best_model,
            'scenario_results': scenario_results
        }
    
    def save_evaluation_results(self, filepath):
        """Save evaluation results to file"""
        results = {
            'evaluation_results': self.evaluation_results,
            'test_scenarios': self.test_scenarios,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Evaluation results saved to {filepath}")
    
    def run_full_evaluation(self, X_test, y_test):
        """Run complete evaluation pipeline"""
        print("Starting full model evaluation...")
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("No models found to evaluate")
            return
        
        # Evaluate each model
        for model_name in self.models:
            print(f"\nEvaluating {model_name}...")
            self.evaluate_model_performance(X_test, y_test, model_name)
        
        # Generate report
        report = self.generate_evaluation_report()
        
        # Save results
        results_path = MODELS_DIR / "evaluation_results.pkl"
        self.save_evaluation_results(results_path)
        
        return report


def main():
    """Main evaluation function"""
    print("Flight Delay Prediction - Model Evaluation")
    print("="*40)
    
    # Create sample data for evaluation
    flight_df, weather_df = create_sample_data()
    
    # Process data
    preprocessor = FlightDataPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(flight_df, weather_df)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Run evaluation
    report = evaluator.run_full_evaluation(X, y)
    
    print(f"\nEvaluation completed!")
    print(f"Best model: {report['best_model']}")


if __name__ == "__main__":
    main()