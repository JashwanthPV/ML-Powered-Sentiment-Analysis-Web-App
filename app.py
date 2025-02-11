import os
import json
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Ensure model directory exists
model_path = "model/sentiment_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

# Load the sentiment model
with open(model_path, "rb") as f:
    model_pipeline = pickle.load(f)

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        user_input = request.form.get("text_input", "").strip()
        if user_input:
            # Predict sentiment
            prediction = model_pipeline.predict([user_input])[0]
            sentiment = "Positive" if prediction else "Negative"
            result = f"Sentiment: {sentiment}"

            # Save entry to JSON file
            data_entry = {"text": user_input, "sentiment": sentiment}
            with open("data.json", "a") as json_file:
                json.dump(data_entry, json_file)
                json_file.write("\n")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
