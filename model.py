import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Load and preprocess data
df = pd.read_csv("dataset/accidents.csv")

# Add synthetic weather data for training (to match real-time weather input)
weather_options = ["Clear", "Rain", "Clouds", "Mist", "Snow", "Fog"]
df["weather"] = np.random.choice(weather_options, size=len(df))

# Encode categorical variables
le = LabelEncoder()
for column in ["Start_Location", "Destination_Location", "Vehicle_Type", "Gender", "weather", "Accident_Severity"]:
    df[column] = le.fit_transform(df[column])

X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "accident_predictor_model.pkl")
print("Model trained and saved as 'accident_predictor_model.pkl'")

# Prediction function
def predict_accident(data, weather):
    model = joblib.load("accident_predictor_model.pkl")
    le = LabelEncoder()
    df = pd.read_csv("dataset/accidents.csv")
    df["weather"] = np.random.choice(weather_options, size=len(df))  # Temporary for encoding consistency
    
    # Encode all features consistently with training data
    for key in ["Start_Location", "Destination_Location", "Vehicle_Type", "Gender", "weather"]:
        le.fit(df[key])
        data[key] = le.transform([data[key]])[0]
    
    # Prepare input DataFrame for model prediction
    input_df = pd.DataFrame([data])
    
    # Rule-based override: Speed > 100, Vehicle_Type = Truck or Bike, Age < 20 or > 50
    le_vehicle = LabelEncoder()
    le_vehicle.fit(df["Vehicle_Type"])
    truck_encoded = le_vehicle.transform(["Truck"])[0]
    bike_encoded = le_vehicle.transform(["Bike"])[0]
    
    if (float(data["Speed"]) > 100 and 
        data["Vehicle_Type"] in [truck_encoded, bike_encoded] and 
        (int(data["Driver_Age"]) < 30 or int(data["Driver_Age"]) > 50)):
        print("Rule triggered: Predicting Severe Accident")
        return "Severe Accident"
    
    # Otherwise, predict using the model based on historical data
    prediction = model.predict(input_df)[0]
    severity_labels = {0: "No Accident", 1: "Slight Chance", 2: "Severe Accident"}
    print(f"Model Prediction based on historical data: {severity_labels[prediction]}")
    return severity_labels[prediction]