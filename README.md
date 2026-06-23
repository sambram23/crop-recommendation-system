
# 🌾 Crop Recommendation System

This is a modern, animated, and interactive web application built with Flask that helps predict the best crop to grow based on various environmental features like soil nutrients, temperature, humidity, pH, and rainfall. It uses multiple machine learning models and provides an insights dashboard with interactive charts.

---

## 🚀 Features

- 🔍 Predict crop using models like KNN, SVM (Linear, RBF, Poly), Decision Tree, Random Forest, and Gradient Boosting.
- 📊 View accuracy and confusion matrix for each model.
- 🌱 Visualize data distributions and relationships (violin plots, bubble charts, heatmaps).
- 🧠 Crop insights with images and reasons for prediction.
- ✨ Futuristic UI with TailwindCSS and Animate.css.

---

## 🧩 Project Structure

```
├── app.py                      # Main Flask application
├── crop_info.py               # Crop details with image and description
├── train_model.py             # Model training script
├── generate_insights.py       # Data visualization + insight generation
├── requirements.txt           # Python dependencies
├── model/                     # Saved ML models and scaler
├── data/
│   └── data.csv               # Dataset used for training and insights
├── static/
│   ├── charts/                # Generated HTML/PNG visual charts
│   └── images/                # Crop images for prediction display
└── templates/
    ├── home.html              # Landing page
    ├── predict.html           # Prediction form & results
    └── models.html            # Models comparison + insights dashboard
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone <https://github.com/rahulgowdaa/crop-recommendation-app.git>
cd crop-app
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Generate models and insights**

```bash
python train_model.py
python generate_insights.py
```

5. **Run the Flask app**

```bash
python app.py
```

6. **Open in your browser**
```
http://localhost:5000
```

---

## 📌 Tech Stack

- Python + Flask
- Tailwind CSS + Animate.css
- scikit-learn + pandas + matplotlib + seaborn + plotly

---

## 🙌 Author

Developed with ❤️ for modern agriculture and data science education.

---

## 📜 License

This project is open-source and available under the MIT License.
