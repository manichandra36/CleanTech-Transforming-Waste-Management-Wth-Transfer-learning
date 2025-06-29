# ♻️ CleanTech: Waste Classifier

CleanTech is an AI-powered waste classification system designed to support smarter, greener waste management. By leveraging transfer learning with TensorFlow, this application can identify whether waste is **Healthy (Compostable)** or **Rotten (Non-Compostable)** using image classification.

## 🚀 Project Features

- 🌿 Classifies waste into compostable or non-compostable categories
- 🤖 Built using transfer learning (TensorFlow/Keras)
- 🔥 Live prediction using a Flask web application
- 🖼️ Simple drag-and-drop image upload interface
- 💡 Future-ready for integration with Smart City IoT systems

---

## 🛠️ Tech Stack!

yaml
Copy code


- **Frontend:** HTML, CSS (in `templates/` and `static/`)
- **Backend:** Python (Flask)
- **Model:** TensorFlow (Pretrained MobileNetV2 or similar)
- **Deployment Ready:** Easily extendable for web hosting

---

## 📁 Project Structure

CleanTech_Waste_Classifier/
│
├── app.py # Flask backend
├── healthy_vs_rotten.h5 # Trained ML model
├── train_model.py # Model training script
├── visualize_data.py # EDA/Visualization
├── /templates/
│ ├── index.html # Main UI
│ ├── about.html
│ └── contact.html
├── /static/
│ └── style.css # CSS styles
├── /dataset/ # Waste image data
│ ├── healthy/
│ └── rotten/
└── README.md # You are here!


---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/manichandra36/CleanTech_Waste_Classifier.git
   cd CleanTech_Waste_Classifier](https://github.com/manichandra36/CleanTech-Transforming-Waste-Management-Wth-Transfer-learning
Create a virtual environment:

bash
Copy code
python -m venv venv
venv\Scripts\activate  # For Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Flask app:

bash
Copy code
python app.py
Open in browser:
Navigate to http://127.0.0.1:5000 in your browser.
