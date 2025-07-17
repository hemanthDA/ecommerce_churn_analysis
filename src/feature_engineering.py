import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_processed_data(self):
        return pd.read_csv('data/processed/customer_features.csv')
        
    def create_rfm_features(self, df):
        """Create RFM (Recency, Frequency, Monetary) features"""
        # RFM scores (1-5 scale)
        df['recency_score'] = pd.qcut(df['days_since_last_purchase'], 5, labels=[5,4,3,2,1])
        df['frequency_score'] = pd.qcut(df['total_orders'].rank(method='first'), 5, labels=[1,2,3,4,5])
        df['monetary_score'] = pd.qcut(df['total_spent'], 5, labels=[1,2,3,4,5])
        
        # Combined RFM score
        df['rfm_score'] = (df['recency_score'].astype(int) + 
                          df['frequency_score'].astype(int) + 
                          df['monetary_score'].astype(int))
        return df
        
    def create_behavioral_features(self, df):
        """Create behavioral features"""
        # Spending patterns
        df['spending_consistency'] = df['total_spent'] / (df['spending_std'] + 1)
        df['avg_days_between_orders'] = df['customer_lifetime_days'] / (df['total_orders'] + 1)
        df['discount_dependency'] = df['total_discount_used'] / (df['total_spent'] + 1)
        
        # Engagement features
        df['orders_per_month'] = df['total_orders'] / (df['account_age_days'] / 30 + 1)
        df['support_intensity'] = df['total_support_tickets'] / (df['account_age_days'] + 1) * 365
        
        return df
        
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_cols = ['gender', 'city', 'customer_segment', 'preferred_category', 
                           'most_frequent_category', 'preferred_payment_method']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df
        
    def select_features(self, df, target_col='is_churned', k=30):
        """Select top k features"""
        # Separate features and target
        X = df.drop([target_col, 'customer_id', 'account_creation_date', 
                    'first_purchase_date', 'last_purchase_date', 'last_ticket_date'], 
                   axis=1, errors='ignore')
        y = df[target_col]
        
        # Select best features
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create final dataset
        final_df = pd.DataFrame(X_selected, columns=selected_features)
        final_df['customer_id'] = df['customer_id'].values
        final_df[target_col] = y.values
        
        return final_df, selected_features
        
    def scale_features(self, df, target_col='is_churned'):
        """Scale numerical features"""
        feature_cols = [col for col in df.columns if col not in ['customer_id', target_col]]
        
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df
        
    def run_feature_engineering(self):
        """Run complete feature engineering pipeline"""
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 30)
        
        # Load data
        df = self.load_processed_data()
        print(f"Loaded: {df.shape}")
        
        # Create features
        df = self.create_rfm_features(df)
        df = self.create_behavioral_features(df)
        df = self.encode_categorical_features(df)
        
        # Select best features
        df_selected, selected_features = self.select_features(df)
        print(f"Selected features: {len(selected_features)}")
        
        # Scale features
        df_final = self.scale_features(df_selected)
        
        # Save
        df_final.to_csv('data/processed/features_final.csv', index=False)
        
        # Save feature list
        pd.DataFrame({'feature': selected_features}).to_csv('data/processed/selected_features.csv', index=False)
        
        print(f"✅ Final dataset: {df_final.shape}")
        print(f"Churn rate: {df_final['is_churned'].mean():.2%}")
        
        return df_final

def main():
    engineer = FeatureEngineer()
    final_data = engineer.run_feature_engineering()

if __name__ == "__main__":
    main()
