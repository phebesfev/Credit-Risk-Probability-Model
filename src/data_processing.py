 # Script for feature engineering
 
import pandas as pd
df = pd.read_csv("../data/raw/data.csv")

# Extract Features
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'] )

df['TransactionHour'] = df['TransactionStartTime'].dt.hour
df['TransactionDay'] = df['TransactionStartTime'].dt.day
df['TransactionMonth'] = df['TransactionStartTime'].dt.month
df['TransactionYear'] = df['TransactionStartTime'].dt.year
# print(df['TransactionStartTime'].head(10))

# Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder

df = pd.get_dummies(df, columns=['ProductCategory'],drop_first=True)

label_col = ['ChannelId', 'ProviderId', 'ProductId']

for i in label_col:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
    
    
# Aggregate Features Adding
agg_df = df.groupby('CustomerId').agg(
    total_transaction_amount=('Amount', 'sum'),
    avg_transaction_amount=('Amount', 'mean'),
    transaction_count=('TransactionId', 'count'),
    std_transaction_amount=('Amount', 'std')
).reset_index() 


# Merge aggregated features back into df
df = df.merge(agg_df, on='CustomerId', how='left')


# Standardization
from sklearn.preprocessing import StandardScaler

numeric_cols = [
    'Amount', 
    'Value', 
    'total_transaction_amount', 
    'avg_transaction_amount', 
    'std_transaction_amount'
]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])




## TASK4

#RFM calculation
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).reset_index()

rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# KMEANS clustering

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_summary = rfm.groupby('cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()

# Sort to find worst (least engaged) group
high_risk_cluster = cluster_summary.sort_values(
    ['Recency', 'Frequency', 'Monetary'],
    ascending=[False, True, True]
).iloc[0]['cluster']

rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

df.to_csv("../data/processed/pdata.csv",index=False)







 