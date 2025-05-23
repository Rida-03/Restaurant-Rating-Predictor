from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model/restaurant_rating_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = pd.DataFrame([{
        'reviewsCount': int(data['reviewsCount']),
        'city': data['city'],
        'categoryName': data['categoryName'],
        'rank': int(data['rank']),
        'avg_price': float((int(data['price_min']) + int(data['price_max'])) / 2)
    }])
    prediction = model.predict(input_data)[0]
    return render_template('index.html', prediction_text=f'Predicted Rating: {prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)