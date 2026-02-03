import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

# Optional: load scaler if needed
# scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Input data:", data)

    # Convert input data to numpy array
    new_data = np.array(list(data.values())).reshape(1, -1)

    # Apply scaling if needed
    # new_data = scalar.transform(new_data)

    output = regmodel.predict(new_data)
    print("Prediction:", output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)