import numpy as np

CLASS_NAMES = [
    "Good",
    "Broken Large",
    "Broken Small",
    "Contamination"
]


def predict_image(model, image, transform):
    input_array = transform(image)
    index, probabilities = model.predict(input_array)

    return (
        CLASS_NAMES[index],
        float(probabilities[index] * 100.0),
        index,
        probabilities
    )
