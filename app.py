from flask import Flask, render_template, request
import requests
from model import predict_accident
import pandas as pd

app = Flask(__name__)
WEATHER_API_KEY = "4343e139816a4cbd89764242252502"

# Load dataset to get unique locations
df = pd.read_csv("dataset/accidents.csv")
unique_locations = sorted(set(df["Start_Location"].tolist() + df["Destination_Location"].tolist()))

def get_weather(location):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["weather"][0]["main"]
        return "Clear"
    except (requests.ConnectionError, requests.Timeout):
        return "Clear"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    precautions = None
    if request.method == "POST":
        weather = get_weather(request.form["start_location"])
        data = {
            "Start_Location": request.form["start_location"],
            "Destination_Location": request.form["destination_location"],
            "Speed": float(request.form["speed"]),
            "Vehicle_Type": request.form["vehicle_type"],
            "Driver_Age": int(request.form["driver_age"]),
            "Gender": request.form["gender"],
            "weather": weather
        }
        prediction = predict_accident(data, weather)
        
        # Precautions based on prediction
        if prediction == "Severe Accident":
            precautions = "Drive slowly, maintain distance, check vehicle condition."
        elif prediction == "Slight Chance":
            precautions = "Be cautious, especially in turns and intersections."
        else:
            precautions = "Drive safely and follow traffic rules."

    return render_template("index.html", prediction=prediction, precautions=precautions, locations=unique_locations)

if __name__ == "__main__":
    app.run(debug=True, port=5003)