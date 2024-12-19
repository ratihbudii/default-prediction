from flask import Flask, jsonify, request
import pickle
import pandas as pd

# Create a Flask app (initialization)
app = Flask(__name__)

# Get the model
model = pickle.load(open('gb-final-calibrated-079.sav', 'rb'))

@app.route('/')
def welcome():
    return "Welcome to the API endpoint!"

@app.route('/test')
def test():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
     # Parse input JSON
    input_data = request.get_json()
    input_data = input_data.get('data')

    if not input_data:
        return jsonify({"error": "Features are required"}), 400
    
    feature_name = model.estimator.named_steps['FeatureEngineering'].feature_names_in_
    df = pd.DataFrame(input_data, columns=feature_name)

    # Perform prediction
    try:
        score = model.predict_proba(df)
        score = score[:,1]
        score = list(score)
        return jsonify({"prediction": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app locally for testing
if __name__ == '__main__':
    app.run(host:'0.0.0.0', port=5000)
