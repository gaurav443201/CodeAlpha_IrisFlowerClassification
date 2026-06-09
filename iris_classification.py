import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load the dataset
print("Loading data...")
csv_path = os.path.join(base_dir, "Iris.csv")
df = pd.read_csv(csv_path)

# Remove the 'Id' column since it is not a feature
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

print("Dataset preview:")
print(df.head())

# 2. Visualize the data
print("Generating plots...")
# Pairplot to see feature relationships
sns.pairplot(df, hue="Species")
plt.title("Iris Feature Pairplot", y=1.02)
plt.savefig(os.path.join(base_dir, "pairplot.png"))
plt.close()

# Boxplot of Petal Length by Species
plt.figure(figsize=(8, 6))
sns.boxplot(x="Species", y="PetalLengthCm", data=df)
plt.title("Petal Length by Species")
plt.savefig(os.path.join(base_dir, "feature_boxplots.png"))
plt.close()

# 3. Split data into features (X) and labels (y)
X = df.drop(columns=['Species'])
y = df['Species']

# 80% train and 20% test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Support Vector Machine (SVM) model
print("Training the SVM model...")
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(base_dir, "confusion_matrix.png"))
plt.close()
print("Confusion matrix saved as confusion_matrix.png")

# 7. Make a sample prediction
sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]]) # Sample measurements
sample_scaled = scaler.transform(sample_flower)
prediction = model.predict(sample_scaled)
print(f"\nSample Prediction for measurements [5.1, 3.5, 1.4, 0.2]: {prediction[0]}")
