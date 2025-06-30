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

df.to_csv("../data/processed/pdata.csv",index=False)


    
    

 