
import os
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Setup
DATA_PATH = "data/data.csv"
MODEL_DIR = "model"
CHART_DIR = "static/charts"
HTML_DIR = "static/charts"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df['label'] = df['label'].astype('category')
df['target'] = df['label'].cat.codes
target_mapping = dict(enumerate(df['label'].cat.categories))

# Train/test
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scaler
scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
X_test_scaled = scaler.transform(X_test)

# Model files
model_files = {
    'knn': 'knn.pkl',
    'svm_linear': 'svm_linear.pkl',
    'svm_rbf': 'svm_rbf.pkl',
    'svm_poly': 'svm_poly.pkl',
    'decision_tree': 'decision_tree.pkl',
    'random_forest': 'random_forest.pkl',
    'gradient_boosting': 'gradient_boosting.pkl'
}

insights = []

# Evaluate each model
for name, file in model_files.items():
    model_path = os.path.join(MODEL_DIR, file)
    if not os.path.exists(model_path):
        continue

    model = pickle.load(open(model_path, 'rb'))
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=target_mapping.values(), columns=target_mapping.values())

    cm_file = f"{name}_confusion.png"
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, cm_file))
    plt.close()

    insights.append({
        "name": name,
        "accuracy": f"{acc * 100:.2f}%",
        "details": f"{name.replace('_', ' ').title()} model evaluated on scaled test set.",
        "confusion": cm_file,
        "chart": None
    })

# Save insights JSON
with open(os.path.join(CHART_DIR, "model_insights.json"), "w") as f:
    json.dump(insights, f, indent=2)

# INTERACTIVE DATA INSIGHTS with Plotly

# 1. Bubble chart
fig1 = px.scatter(df, x='rainfall', y='humidity', color='label',
                  size='temperature', hover_name='label', title="🌧️ Rainfall vs Humidity Bubble Chart")
pio.write_html(fig1, file=os.path.join(HTML_DIR, "bubble_chart.html"), full_html=False)

# 2. Donut chart
label_counts = df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']
fig2 = px.pie(label_counts, names='label', values='count', hole=0.4, title="🌱 Crop Distribution")
pio.write_html(fig2, file=os.path.join(HTML_DIR, "donut_chart.html"), full_html=False)

# 3. Heatmap: Temperature vs pH
heatmap_data = df[['temperature', 'ph', 'rainfall']]
heatmap_data = heatmap_data.groupby(['temperature', 'ph'], as_index=False).mean()
heatmap_pivot = heatmap_data.pivot(index='temperature', columns='ph', values='rainfall')
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_pivot, cmap='YlOrRd', annot=False)
plt.title("Temperature vs pH Heatmap (Rainfall-based)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "temp_ph_heatmap.png"))
plt.close()

# 4. Violin plots: Feature distribution by crop
for col in ['N','P','K','ph','temperature']:
    fig = px.violin(df, y=col, x='label', box=True, points='all', title=f"🎻 {col.upper()} Distribution by Crop")
    filename = f"{col.upper()}_violin.html" if col != 'ph' else "ph_violin.html"
    pio.write_html(fig, file=os.path.join(HTML_DIR, filename), full_html=False)

print("✅ All models, insights, and visualizations updated.")
