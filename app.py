from flask import Flask, render_template, request
from PIL import Image
import io
import base64
import os

from utils.model_loader import load_model
from utils.preprocessing import get_transforms
from utils.predictor import predict_image
from utils.gradcam import generate_gradcam

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "resnet18_fused.npz")

# Lightweight NumPy model; no PyTorch required at runtime.
model = load_model(MODEL_PATH)
transform = get_transforms()


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]

    if file.filename == "":
        return "No selected file", 400

    try:
        image = Image.open(file).convert("RGB")
    except Exception as e:
        return f"Invalid or corrupted image: {str(e)}", 400

    try:
        prediction, confidence, _, _ = predict_image(
            model, image, transform
        )

        explanation = generate_gradcam(
            model, image, transform
        )

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=round(confidence, 2),
            uploaded_image=image_to_data_url(image),
            gradcam_image=image_to_data_url(
                Image.fromarray(explanation)
            )
        )

    except Exception as e:
        return f"Error during model inference: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
