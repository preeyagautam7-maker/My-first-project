from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("crop_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    data = [[N, P, K, temperature, humidity, ph, rainfall]]

    prediction = model.predict(data)

    crop = prediction[0]

    crop_info = {
        "rice": "Rice grows well in warm and humid conditions with sufficient rainfall.",
        "maize": "Maize requires warm temperature, moderate rainfall and fertile soil.",
        "chickpea": "Chickpea grows well in cool and relatively dry conditions.",
        "kidneybeans": "Kidney beans require moderate temperature and well-drained soil.",
        "pigeonpeas": "Pigeon peas grow well in warm climates with moderate rainfall.",
        "mothbeans": "Moth beans are suitable for warm and dry conditions.",
        "mungbean": "Mung bean grows well in warm weather and moderate rainfall.",
        "blackgram": "Black gram grows well in warm and humid conditions.",
        "lentil": "Lentil prefers cool weather and well-drained soil.",
        "banana": "Banana requires warm temperature, high humidity and sufficient water.",
        "mango": "Mango grows well in warm climates with suitable rainfall.",
        "grapes": "Grapes prefer warm weather and relatively dry conditions.",
        "watermelon": "Watermelon requires warm temperature and adequate moisture.",
        "papaya": "Papaya grows well in warm and humid conditions.",
        "coconut": "Coconut grows best in warm, humid and tropical climates."
    }

    info = crop_info.get(
        crop.lower(),
        "This crop is suitable based on the soil and environmental conditions provided."
    )

    return render_template(
        "index.html",
        prediction=crop,
        crop_info=info
    )


if __name__ == "__main__":
    app.run(debug=True)