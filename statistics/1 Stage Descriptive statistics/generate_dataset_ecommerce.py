import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Simulate realistic e-commerce order data
customer_ids = [f"CUST_{i:04d}" for i in range(1, n+1)]
cities = np.random.choice(['Chennai', 'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Pune', 'Kolkata'], n, p=[0.2,0.18,0.17,0.15,0.13,0.1,0.07])
categories = np.random.choice(['Electronics', 'Fashion', 'Groceries', 'Books', 'Home Decor', 'Sports'], n, p=[0.25,0.22,0.18,0.12,0.13,0.10])

# Order values — realistic right-skewed distribution
base_order = np.random.lognormal(mean=4.5, sigma=0.8, size=n)
# Add category multipliers
cat_mult = {'Electronics':3.5, 'Fashion':1.2, 'Groceries':0.6, 'Books':0.5, 'Home Decor':1.8, 'Sports':1.4}
order_values = np.array([base_order[i] * cat_mult[categories[i]] for i in range(n)])
order_values = np.clip(order_values, 50, 50000).round(2)

# Add some big spenders (outliers — 2% of users)
outlier_idx = np.random.choice(n, size=20, replace=False)
order_values[outlier_idx] = np.random.uniform(15000, 50000, 20).round(2)

# Age groups
ages = np.random.choice(['18-25','26-35','36-45','46-55','55+'], n, p=[0.22,0.35,0.25,0.12,0.06])

# Number of items
num_items = np.random.randint(1, 12, n)

# Days since last order
recency_days = np.random.exponential(scale=45, size=n).astype(int).clip(1, 365)

# Rating (1-5)
ratings = np.random.choice([1,2,3,4,5], n, p=[0.04,0.07,0.16,0.40,0.33])

# Discount applied
discount_pct = np.random.choice([0, 5, 10, 15, 20, 25, 30], n, p=[0.30,0.15,0.20,0.15,0.10,0.06,0.04])

df = pd.DataFrame({
    'customer_id': customer_ids,
    'city': cities,
    'age_group': ages,
    'category': categories,
    'order_value_inr': order_values,
    'num_items': num_items,
    'recency_days': recency_days,
    'rating': ratings,
    'discount_pct': discount_pct
})

df.to_csv('/home/claude/descriptive_stats/ecommerce_orders.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())
print(df.dtypes)
