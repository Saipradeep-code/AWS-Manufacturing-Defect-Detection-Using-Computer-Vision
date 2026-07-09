import torch

CLASS_NAMES = [
    "Good",
    "Broken Large",
    "Broken Small",
    "Contamination"
]

def predict_image(model, image, transform):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

        print("Raw probabilities:", probabilities)
        print("Predicted index:", predicted.item())

    return (
        CLASS_NAMES[predicted.item()],
        confidence.item() * 100
    )