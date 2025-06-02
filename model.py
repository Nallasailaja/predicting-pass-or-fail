import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    'Previous Scores': [80, 45, 70, 30, 90, 60, 65, 40, 75, 50],
    'Performance Index': [85, 40, 70, 30, 90, 60, 65, 45, 75, 55]
}

df = pd.DataFrame(data)

X = df[['Previous Scores']]
y = df['Performance Index']

model = LinearRegression()
model.fit(X, y)

# Save model with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
