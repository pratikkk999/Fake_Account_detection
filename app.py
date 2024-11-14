from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained models
with open('random_forest_fake_account_classifier.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('logistic_regression_fake_account_classifier.pkl', 'rb') as lr_file:
    lr_model = pickle.load(lr_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    input_data = [
        int(form_data['username_length']),
        int(form_data['username_has_number']),
        int(form_data['full_name_has_number']),
        int(form_data['full_name_length']),
        int(form_data['is_private']),
        int(form_data['is_joined_recently']),
        int(form_data['has_channel']),
        int(form_data['is_business_account']),
        int(form_data['has_guides']),
        int(form_data['has_external_url']),
    ]
    
    # Get predictions from both models
    rf_prediction = rf_model.predict([input_data])[0]
    lr_prediction = lr_model.predict([input_data])[0]

    return render_template(
        'index.html',
        rf_result="Fake Account" if rf_prediction == 1 else "Not Fake Account",
        lr_result="Fake Account" if lr_prediction == 1 else "Not Fake Account"
    )

if __name__ == '__main__':
    app.run(debug=True)
