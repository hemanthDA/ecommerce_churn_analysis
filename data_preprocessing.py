import pandas as pd
import os


class DataPreprocessor:
    def __init__(self):
        self.customers_df = None
        self.transactions_df = None
        self.processed_df = None

    def load_data(self):
        print("📥 Loading raw data...")
        self.customers_df = pd.read_csv('data/raw/customers.csv')
        self.transactions_df = pd.read_csv('data/raw/transactions.csv')
        print("✅ Raw data loaded successfully.")
        print(f"Customers: {self.customers_df.shape}, Transactions: {self.transactions_df.shape}")

    def preprocess_transactions(self):
        print("🔄 Preprocessing transactions...")
        self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])

        # Aggregate transaction data by customer
        tx_agg = self.transactions_df.groupby('customer_id').agg({
            'amount': ['count', 'sum', 'mean', 'max'],
            'transaction_date': ['min', 'max']
        })

        # Rename columns
        tx_agg.columns = ['txn_count', 'total_spent_txn', 'avg_spent_txn', 'max_spent_txn', 'first_txn_date', 'last_txn_date']
        tx_agg.reset_index(inplace=True)

        print(f"🧾 Aggregated transaction features: {tx_agg.shape}")
        return tx_agg

    def merge_data(self, tx_agg):
        print("🔗 Merging customer data with transaction features...")
        df = pd.merge(self.customers_df, tx_agg, on='customer_id', how='left')

        # Fill missing values for customers with no transactions
        df[['txn_count', 'total_spent_txn', 'avg_spent_txn', 'max_spent_txn']] = df[[
            'txn_count', 'total_spent_txn', 'avg_spent_txn', 'max_spent_txn']].fillna(0)

        df['first_txn_date'] = pd.to_datetime(df['first_txn_date'])
        df['last_txn_date'] = pd.to_datetime(df['last_txn_date'])

        df['days_since_last_txn'] = (pd.to_datetime('today') - df['last_txn_date']).dt.days.fillna(-1)

        self.processed_df = df
        print(f"✅ Merged dataset shape: {df.shape}")

    def save_processed_data(self):
        os.makedirs("data/processed", exist_ok=True)
        self.processed_df.to_csv("data/processed/customer_features.csv", index=False)
        print("💾 Processed data saved to data/processed/customer_features.csv")

    def run_preprocessing(self):
        self.load_data()
        tx_agg = self.preprocess_transactions()
        self.merge_data(tx_agg)
        self.save_processed_data()
        return self.processed_df


def main():
    print("🚀 STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_preprocessing()
    print("🎉 DATA PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
