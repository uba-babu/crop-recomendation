from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model and scalers
model_path = r"C:\Users\U.B.A Yadav\OneDrive\Desktop\crop\venv\model (1).pkl"
stand_scaler_path = r"C:\Users\U.B.A Yadav\OneDrive\Desktop\crop\venv\standscaler (3).pkl"
minmax_scaler_path = r"C:\Users\U.B.A Yadav\OneDrive\Desktop\crop\venv\minmaxscaler (2).pkl"

with open(model_path, 'rb') as model_file, \
     open(stand_scaler_path, 'rb') as stand_scaler_file, \
     open(minmax_scaler_path, 'rb') as minmax_scaler_file:
    model = pickle.load(model_file)
    stand_scaler = pickle.load(stand_scaler_file)
    minmax_scaler = pickle.load(minmax_scaler_file)

# Create Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Convert form data to numpy array
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale features
    scaled_features = minmax_scaler.transform(stand_scaler.transform(single_pred))

    # Make prediction
    prediction = model.predict(scaled_features)

    # Mapping prediction to crop name
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Get the predicted crop name
    crop = crop_dict.get(prediction[0], "Unknown")

    # Prepare result message
    if crop != "Unknown":
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Render the template with the result message
    return render_template('index.html', result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
