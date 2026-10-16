import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/reviews_prepared.csv")
labels = df["sentiment"].map({"negative": 0, "positive": 1}).to_numpy()

_, test_text, _, test_labels = train_test_split(
    df["Text"].astype(str).tolist(),
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

model = tf.keras.models.load_model("sentiment_model.keras")
probabilities = model.predict(
    np.array(test_text, dtype=object), verbose=1
).ravel()
predictions = (probabilities >= 0.5).astype(int)

report = classification_report(
    test_labels,
    predictions,
    target_names=["Negative", "Positive"],
)
print(report)

with open("classification_report.txt", "w") as file:
    file.write(report)

matrix = confusion_matrix(test_labels, predictions)
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted sentiment")
plt.ylabel("Actual sentiment")
plt.title("Product Review Sentiment — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved classification_report.txt and confusion_matrix.png")