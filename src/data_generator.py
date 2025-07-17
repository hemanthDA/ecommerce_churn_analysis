import pandas as pd
from faker import Faker
import random
import os

fake = Faker()

def generate_fake_customers(num_records=500):
    """Generate fake customer data."""
    data = []
    for _ in range(num_records):
        data.append({
            'customer_id': fake.uuid4(),
            'name': fake.name(),
            'email': fake.email(),
            'gender': random.choice(['Male', 'Female', 'Other']),
            'age': random.randint(18, 70),
            'country': fake.country(),
            'signup_date': fake.date_between(start_date='-2y', end_date='-6mo'),
            'last_purchase_date': fake.date_between(start_date='-6mo', end_date='today'),
            'purchase_count': random.randint(0, 50),
            'total_spent': round(random.uniform(10, 5000), 2),
            'churned': random.choice([0, 1])
        })

    df = pd.DataFrame(data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/customers.csv", index=False)
    print("✅ customers.csv saved to data/raw/")
    return df


def generate_fake_transactions(num_records=1500, customer_ids=None):
    """Generate fake transaction data for customers."""
    if customer_ids is None:
        customer_ids = [fake.uuid4() for _ in range(100)]

    data = []
    for _ in range(num_records):
        data.append({
            'transaction_id': fake.uuid4(),
            'customer_id': random.choice(customer_ids),
            'transaction_date': fake.date_between(start_date='-1y', end_date='today'),
            'amount': round(random.uniform(5.0, 1500.0), 2),
            'category': random.choice(['Electronics', 'Clothing', 'Groceries', 'Books', 'Home', 'Others'])
        })

    df = pd.DataFrame(data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/transactions.csv", index=False)
    print("✅ transactions.csv saved to data/raw/")
    return df


if __name__ == "__main__":
    print("🚀 Generating mock customer and transaction data...")
    customer_df = generate_fake_customers(num_records=500)
    transaction_df = generate_fake_transactions(num_records=1500, customer_ids=customer_df['customer_id'].tolist())
    print("🎉 Data generation complete!")
