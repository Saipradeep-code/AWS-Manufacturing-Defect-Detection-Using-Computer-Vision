from flask import Flask, render_template, request
from PIL import Image
import os
from werkzeug.utils import secure_filename

from utils.model_loader import load_model
from utils.preprocessing import get_transforms
from utils.predictor import predict_image
from utils.gradcam import generate_gradcam

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

model = load_model("model/best_resnet18_bottle.pth")
transform = get_transforms()


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

    filename = secure_filename(file.filename)
    if not filename:
        filename = "uploaded_image.png"

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        image = Image.open(file).convert("RGB")
        image.save(upload_path)
    except Exception as e:
        return f"Invalid or corrupted image: {str(e)}", 400

    try:
        prediction, confidence = predict_image(
            model,
            image,
            transform
        )

        gradcam = generate_gradcam(
            model,
            image,
            transform
        )

        gradcam_filename = f"gradcam_{filename}"
        if not gradcam_filename.lower().endswith('.png'):
            gradcam_filename = os.path.splitext(gradcam_filename)[0] + '.png'
            
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
        Image.fromarray(gradcam).save(gradcam_path)

        # Normalize paths for URLs (Windows compatible slashes)
        uploaded_image_url = upload_path.replace("\\", "/")
        gradcam_image_url = gradcam_path.replace("\\", "/")

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