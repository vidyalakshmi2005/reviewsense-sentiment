# ReviewSense

ReviewSense is a Windows desktop sentiment analyzer. It opens in its own window, accepts free text, and estimates whether the text is positive or negative using a TensorFlow LSTM model.

The model always chooses one of two labels. It learned from product reviews, so results on unrelated, neutral, or mixed text may be less reliable.

## Start the desktop app

Open Command Prompt and run:

```cmd
D:\reviewsense-sentiment-main\run_local.bat
```

The first launch creates a Python environment and installs TensorFlow if needed; this can take several minutes. Later launches open the desktop window directly. To close the app, close its window.

## Main files

- `desktop_app.py` — native Windows interface and sentiment prediction
- `sentiment_model.keras` — trained LSTM model
- `data/reviews_prepared.csv` — example dataset for exploration
- `app.py` — previous Streamlit web interface; the desktop launcher does not use it
