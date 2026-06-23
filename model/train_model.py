
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(os.path.join('data', 'data.csv'))

# Encode target
c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes

# Save mapping
with open('model/target_mapping.pkl', 'wb') as f:
    pickle.dump(targets, f)

# Features and target
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

model_scores = {}

# Train and save each model
models = {
    'knn': KNeighborsClassifier(),
    'svm_linear': SVC(kernel='linear'),
    'svm_rbf': SVC(kernel='rbf'),
    'svm_poly': SVC(kernel='poly'),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'random_forest': RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    model_scores[name] = acc
    with open(f'model/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ {name} accuracy: {acc:.4f}")

# Final reporting
best_model = max(model_scores, key=model_scores.get)
print(f"\n🏆 Best model is '{best_model}' with accuracy: {model_scores[best_model]:.4f}")
