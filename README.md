# AWS Manufacturing Defect Detection using Computer Vision

## Overview

AWS Manufacturing Defect Detection using Computer Vision is an AI-powered quality inspection system designed to automatically identify manufacturing defects in bottle products using Deep Learning and Computer Vision.

The application is built using PyTorch, Flask, and a fine-tuned ResNet18 model. It provides real-time defect classification along with Grad-CAM based visual explanations to improve transparency and interpretability of model predictions.

The project follows a cloud-ready architecture and is designed for deployment on Amazon Web Services (AWS), making it suitable for scalable manufacturing inspection systems.

---

## Problem Statement

Traditional quality inspection in manufacturing industries relies heavily on manual inspection, which is often slow, inconsistent, and prone to human error.

This project automates the inspection process using Artificial Intelligence, enabling fast, accurate, and explainable defect detection.

---

## Objectives

- Automate manufacturing quality inspection
- Detect bottle defects using Deep Learning
- Provide explainable predictions using Grad-CAM
- Build a cloud-ready AI application using AWS
- Develop an interactive web interface for real-time inspection

---

## Key Features

- Manufacturing defect detection using Computer Vision
- Fine-tuned ResNet18 deep learning model
- Explainable AI using Grad-CAM
- Real-time image classification
- Confidence score prediction
- Interactive Flask web application
- Modern responsive user interface
- AWS deployment ready
- Cross-platform compatibility

---

## Dataset

Dataset: MVTec AD Bottle Dataset

The model is trained to classify the following categories:

- Good
- Broken Large
- Broken Small
- Contamination

---

## Technology Stack

### Artificial Intelligence

- PyTorch
- Torchvision
- ResNet18
- Grad-CAM
- NumPy
- OpenCV
- Pillow

### Backend

- Flask
- Jinja2

### Frontend

- HTML5
- CSS3
- JavaScript

### Cloud Platform

- Amazon Web Services (AWS)
- Amazon EC2
- AWS Learner Lab
- GitHub

---

## AWS Architecture

```
                User
                  |
                  v
         Flask Web Application
                  |
                  v
        Image Preprocessing
                  |
                  v
         ResNet18 Deep Learning Model
                  |
                  v
          Defect Classification
                  |
                  v
          Grad-CAM Generation
                  |
                  v
          Prediction Results
```

The project is designed to be deployed on AWS EC2 and can be extended with additional AWS services such as:

- Amazon S3
- Amazon SageMaker
- AWS Elastic Beanstalk
- Amazon CloudWatch
- AWS IAM

---

## Project Structure

```text
AWS-Manufacturing-Defect-Detection-Using-Computer-Vision
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model
│   └── best_resnet18_bottle.pth
│
├── templates
│   └── index.html
│
├── static
│   ├── css
│   ├── js
│   ├── images
│   ├── uploads
│   └── gradcam
│
├── utils
│   ├── model_loader.py
│   ├── predictor.py
│   ├── preprocessing.py
│   └── gradcam.py
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Saipradeep-code/AWS-Manufacturing-Defect-Detection-Using-Computer-Vision.git
```

Navigate to the project directory

```bash
cd AWS-Manufacturing-Defect-Detection-Using-Computer-Vision
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install the dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open the application

```
http://127.0.0.1:5000
```

---

## Application Workflow

1. Upload a bottle image.
2. The image is preprocessed.
3. The ResNet18 model performs defect classification.
4. A confidence score is generated.
5. Grad-CAM produces a visual explanation.
6. The final prediction and visualization are displayed.

---

## Model Information

| Parameter | Value |
|----------|-------|
| Model | ResNet18 |
| Framework | PyTorch |
| Explainability | Grad-CAM |
| Dataset | MVTec AD Bottle |
| Classes | 4 |
| Backend | Flask |
| Deployment Platform | AWS EC2 |

---

## AWS Deployment

The application is designed for deployment on Amazon Web Services.

Deployment environment includes:

- Amazon EC2
- Python Virtual Environment
- Flask
- Gunicorn
- Nginx
- GitHub
- Linux (Ubuntu)

The architecture allows future integration with Amazon S3 for image storage and Amazon SageMaker for managed model hosting.

---

## Future Enhancements

- Batch image processing
- Live camera-based inspection
- Manufacturing analytics dashboard
- PDF inspection reports
- REST API
- Docker support
- Amazon S3 integration
- Amazon SageMaker endpoint deployment
- CloudWatch monitoring
- CI/CD using GitHub Actions

---

## Author

Sai Pradeep

GitHub Repository

https://github.com/Saipradeep-code/AWS-Manufacturing-Defect-Detection-Using-Computer-Vision

---

## License

This project was developed for academic learning, research, and educational purposes.
