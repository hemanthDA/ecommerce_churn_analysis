import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import joblib

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.test_data = None
        
    def load_models_and_data(self):
        """Load trained models and test data"""
        # Load models
        self.models['logistic_regression'] = joblib.load('src/models/logistic_regression_model.pkl')
        self.models['random_forest'] = joblib.load('src/models/random_forest_model.pkl')
        
        # Load test data
        df = pd.read_csv('data/processed/features_final.csv')
        from sklearn.model_selection import train_test_split
        
        X = df.drop(['customer_id', 'is_churned'], axis=1)
        y = df['is_churned']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.test_data = (X_test, y_test)
        print("✅ Models and data loaded")
        
    def create_evaluation_plots(self):
        """Create evaluation visualizations"""
        X_test, y_test = self.test_data
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ROC Curves
        ax1 = axes[0, 0]
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = np.trapz(tpr, fpr)
            ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        
        # Precision-Recall Curves
        ax2 = axes[0, 1]
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ax2.plot(recall, precision, label=name)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        
        # Confusion Matrices
        for i, (name, model) in enumerate(self.models.items()):
            ax = axes[1, i]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title(f'{name} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('src/models/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        # Get feature names
        features_df = pd.read_csv('data/processed/selected_features.csv')
        feature_names = features_df['feature'].tolist()
        
        # Random Forest feature importance
        rf_model = self.models['random_forest']
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('src/models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance
        importance_df.to_csv('src/models/feature_importance.csv', index=False)
        
    def generate_model_report(self):
        """Generate comprehensive model report"""
        X_test, y_test = self.test_data
        
        report_data = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1_Score': f1_score(y_test, y_pred),
                'AUC_Score': roc_auc_score(y_test, y_pred_proba)
            }
            
            report_data.append(metrics)
        
        # Create report DataFrame
        report_df = pd.DataFrame(report_data)
        report_df = report_df.round(3)
        
        # Save report
        report_df.to_csv('src/models/model_comparison_report.csv', index=False)
        
        print("MODEL COMPARISON REPORT")
        print("=" * 25)
        print(report_df.to_string(index=False))
        
        # Best model
        best_model = report_df.loc[report_df['AUC_Score'].idxmax(), 'Model']
        print(f"\n🏆 Best Model: {best_model}")
        
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("MODEL EVALUATION PIPELINE")
        print("=" * 30)
        
        self.load_models_and_data()
        self.create_evaluation_plots()
        self.feature_importance_analysis()
        self.generate_model_report()
        
        print(f"\n✅ Evaluation complete. Results saved to src/models/")

def main():
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
