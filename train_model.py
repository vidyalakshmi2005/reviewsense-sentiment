import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Read the prepared reviews
df = pd.read_csv("data/reviews_prepared.csv")

# Convert labels to numbers: negative = 0, positive = 1
labels = df["sentiment"].map({"negative": 0, "positive": 1}).to_numpy()

# Split the data: 80% for training, 20% for testing
train_text, test_text, train_labels, test_labels = train_test_split(
    df["Text"].astype(str).tolist(),
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

# Turn review words into number sequences for the neural network
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20000,
    output_mode="int",
    output_sequence_length=150,
)
vectorizer.adapt(train_text)

# Build an LSTM deep learning model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(), dtype=tf.string),
    vectorizer,
    tf.keras.layers.Embedding(input_dim=20000, output_dim=64, mask_zero=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    np.array(train_text, dtype=object),
    train_labels,
    validation_split=0.1,
    epochs=3,
    batch_size=128,
)

# Check performance on reviews the model did not train on
loss, accuracy = model.evaluate(
    np.array(test_text, dtype=object),
    test_labels,
)
print(f"\nTest accuracy: {accuracy:.2%}")

# Save the trained model for later use
model.save("sentiment_model.keras")
print("Model saved as sentiment_model.keras")