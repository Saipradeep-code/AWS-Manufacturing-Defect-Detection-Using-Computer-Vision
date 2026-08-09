from utils.model_numpy import ResNet18Numpy


def load_model(model_path):
    return ResNet18Numpy(model_path)
