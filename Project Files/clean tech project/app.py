# app.py  ── Flask backend for CleanTech Waste Classifier
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# load the model saved in the modern .keras format
model = load_model("waste_classifier.keras", compile=False)

# where uploaded images will be stored
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # make sure the folder exists

CLASSES = ["Biodegradable", "Recyclable", "Trash"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1) basic checks
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # 2) save the image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # 3) preprocess the image to match training pipeline
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0          # scale 0‑1
    img_array = np.expand_dims(img_array, axis=0)        # batch of 1

    # 4) model prediction
    prediction = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]

    # 5) render result
    return render_template(
        "result.html",
        prediction=predicted_class,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)
