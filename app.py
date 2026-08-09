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
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_resnet18_bottle.pth"
)

# Load model when the server starts
model = load_model(MODEL_PATH)
transform = get_transforms()


def image_to_data_url(image):
    """Convert PIL image to a browser-friendly data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

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

    # Read uploaded image
    try:
        image = Image.open(file).convert("RGB")
    except Exception as e:
        return f"Invalid or corrupted image: {str(e)}", 400

    # Run prediction
    try:

        prediction, confidence = predict_image(
            model,
            image,
            transform
        )

        # Generate Grad-CAM
        gradcam = generate_gradcam(
            model,
            image,
            transform
        )

        gradcam_image = Image.fromarray(gradcam)

        # Convert images to browser-displayable URLs
        uploaded_image_url = image_to_data_url(image)
        gradcam_image_url = image_to_data_url(gradcam_image)

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=round(confidence, 2),
            uploaded_image=uploaded_image_url,
            gradcam_image=gradcam_image_url
        )

    except Exception as e:
        return f"Error during model inference: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)