from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the saved model once on startup
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError("Please run model.py first to create model.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        previous_score = float(request.form['previous_score'])
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter a numeric score.'})

    user_input = pd.DataFrame({'Previous Scores': [previous_score]})
    predicted_index = model.predict(user_input)[0]
    result = "Pass" if predicted_index >= 50 else "Fail"

    return jsonify({
        'predicted_index': round(predicted_index, 2),
        'result': result
    })

if __name__ == "__main__":
    app.run(debug=True)
