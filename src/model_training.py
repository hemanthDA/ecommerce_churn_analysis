import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load processed features"""
        df = pd.read_csv('data/processed/features_final.csv')
        
        X = df.drop(['customer_id', 'is_churned'], axis=1)
        y = df['is_churned']
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    def handle_imbalance(self, X_train, y_train):
        """Handle class imbalance with SMOTE"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original: {len(y_train)}, Resampled: {len(y_resampled)}")
        return X_resampled, y_resampled
        
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression"""
        print("Training Logistic Regression...")
        
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['logistic_regression'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        print("Training Random Forest...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['random_forest'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\nMODEL EVALUATION")
        print("=" * 20)
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            self.results[name] = {
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n{name.upper()}:")
            print(f"AUC Score: {auc_score:.3f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            joblib.dump(model, f'src/models/{name}_model.pkl')
        
        # Save results
        results_df = pd.DataFrame({
            'model': list(self.results.keys()),
            'auc_score': [self.results[model]['auc_score'] for model in self.results.keys()]
        })
        results_df.to_csv('src/models/model_results.csv', index=False)
        
        print(f"\n✅ Models saved to src/models/")
        
    def run_training(self):
        """Run complete training pipeline"""
        print("CHURN MODEL TRAINING")
        print("=" * 25)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        
        # Train models
        self.train_logistic_regression(X_train_balanced, y_train_balanced)
        self.train_random_forest(X_train_balanced, y_train_balanced)
        
        # Evaluate
        self.evaluate_models(X_test, y_test)
        
        # Save
        self.save_models()
        
        return self.models, self.results

def main():
    trainer = ChurnModelTrainer()
    models, results = trainer.run_training()

if __name__ == "__main__":
    main()
