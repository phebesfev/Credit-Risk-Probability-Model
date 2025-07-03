 # Pydantic models for API
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    CountryCode: int
    ProviderId: int
    ProductId: int
    ChannelId: int
    Amount: float
    Value: float
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    PricingStrategy: int
    ProductCategory_data_bundles: int
    ProductCategory_financial_services: int
    ProductCategory_movies: int
    ProductCategory_other: int
    ProductCategory_ticket: int
    ProductCategory_transport: int
    ProductCategory_tv: int
    ProductCategory_utility_bill: int
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float
    is_high_risk: int

class PredictionResponse(BaseModel):
    risk_probability: float

 