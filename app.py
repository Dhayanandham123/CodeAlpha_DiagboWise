from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values as float
        features = [float(x) for x in request.form.values()]
        
        # Ensure the input shape is correct
        prediction = model.predict([features])[0]
        
        # Interpret prediction
        result = " The person does NOT have Heart Disease." if prediction == 0 else "The person has Heart Disease."
        return render_template('result.html', result=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
