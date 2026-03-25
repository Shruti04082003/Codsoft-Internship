import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("advertising.csv")

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("sales_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")