import numpy as np
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(model, image, transform):

    target_layer = model.layer4[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization