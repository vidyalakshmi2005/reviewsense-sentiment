import tensorflow as tf

# Load the model you trained
model = tf.keras.models.load_model("sentiment_model.keras")

while True:
    review = input("\nType a product review (or type 'quit' to stop): ")

    if review.lower() == "quit":
        break

    # The model returns a score between 0 and 1
    score = float(model.predict(tf.constant([review]), verbose=0)[0][0])

    if score >= 0.5:
        sentiment = "Positive"
        confidence = score
    else:
        sentiment = "Negative"
        confidence = 1 - score

    print(f"Prediction: {sentiment} ({confidence:.1%} confidence)")