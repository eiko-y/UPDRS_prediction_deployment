# UPDRS_prediction_deployment
Deployment strategy for a UPDRS prediction model using FastAPI and AWS Sagemaker

## Parkinson’s Disease Progression Prediction Using Voice Measurements
Project Overview
This project leverages 16 biometric voice measurements to predict the total Unified Parkinson’s Disease Rating Scale (total_UPDRS) score, providing a quantitative and non-invasive assessment of Parkinson’s disease progression. By analyzing voice features, this model enables:
1. Patients to remotely monitor their disease progression.


2. Clinicians to continuously track patient status without frequent in-person visits.


3. Healthcare systems to improve accessibility and reduce the need for hospital visits, saving time and resources while maintaining care quality.


This approach supports scalable, remote, and timely interventions for better patient management and health outcomes.

## Tech Stack & Tools
Programming Language: Python 3.9


Model Development & Training: Google Colab, utilizing the UCI Machine Learning Repository dataset


Web Framework: FastAPI (for serving predictions via a RESTful API)


### Core Libraries:


scikit-learn (v1.3.2) for model development and inference


joblib for model serialization


pydantic for input data validation


uvicorn as the ASGI server for FastAPI


### Containerization & Deployment:


Docker for application packaging and dependency management


AWS SageMaker for model hosting, endpoint management, and scalable deployment

## Project Structure Highlights
fastapi-deploy/app/ — Contains the trained model file (model.pkl) and FastAPI application code (main.py)


sagemaker_deploy/ — Holds the SageMaker deployment resources, including:


  * inference.py (inference script for SageMaker endpoint)


*   model.pkl (zipped model artifact)


 *  requirements.txt (SageMaker environment dependencies)
  
## AWS Resources Overview
S3 Bucket: Stores model artifacts (model.pkl) and deployment scripts. The bucket is configured with appropriate permissions for SageMaker to access the model during endpoint creation.


ECR (Elastic Container Registry): Hosts the Docker image containing the FastAPI application and dependencies, enabling easy and repeatable deployment to SageMaker.


SageMaker Endpoint: A fully managed HTTPS endpoint hosting the model for real-time inference. The endpoint name is updrs-endpoint-v5 (update as appropriate), configured to auto-scale based on request load.

## Usage
Running Locally
1. Clone this repository.


2. Create and activate a Python virtual environment.


3. Install dependencies: pip install -r requirements.txt


4. Run the FastAPI app locally with Uvicorn:


* uvicorn main:app --reload --port 8001
* Access Swagger UI at http://127.0.0.1:8001/docs to test the API.


## Deploying on AWS SageMaker
1. Build and push the Docker container image to ECR.


2. Upload model artifacts and inference scripts to the S3 bucket.


3. Create a SageMaker model referencing the container and model data.


4. Configure and deploy the SageMaker endpoint.


5. Query the endpoint for predictions using the FastAPI client or any HTTP client.
   
## Logging and Monitoring
The FastAPI app includes basic logging with timestamps, log levels, and messages for tracking prediction requests and debugging.
