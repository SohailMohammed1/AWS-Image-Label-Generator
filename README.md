# AWS Project 1: Image Label Generator

The following application is a Command Line tool which analyses images stored within Amazon S3 buckets using Amazon Rekognition. 

## Project Summary

The AWS Image Label Generator software is optimised to recognise a diverse range of objects located within an image. It utilises the command line interface for ease of use. The analysed images are saved as .png files. 

For each of the images, the programme: 

- Calls upon AWS Rekognition to detect objects and create labels
- Creates 'confidence' scores which ascertains the labels created 
- Draws 'bounding boxes' around the detected objects and saves an annotated copy locally
- Prints a summary of labels to the terminal

## Architecture Overview

Below are the services and components used:

- Amazon S3: Stores the images in a bucket to be analysed
- Amazon Rekognition: Identifies objects and performs label classification
- Python: `boto3, Pillow, Matplotlib` executes the processing pipeline
- CLI: Terminal point for users to submit images for analysis 

![AWS Architecture](/workspaces/AWS-Image-Label-Generator/Screenshots/Screenshot 2025-12-29 152748.png)

## Workflow Diagram

1. Users input one or more S3 object keys into the CLI
2. Application retrieves S3 object from bucket
3. Application calls on Rekognition service using `DetectLabels`
4. Rekognition returns labels + confidence metrics
5. Application renders bounding boxes
6. Application saves annotated outputs and prints summary locally 

## Technical Process

- User passes one or more S3 image filenames (object keys) in the command line. 
- For each image, the label: 
- Sends out a 'DetectLabels' request to AWS Rekognition (`boto3.client('rekognition')`).
- Logs each label name and confidence score to the terminal
- The image is downloaded using the (`boto3.resource('s3')`)
- Bounding boxes are annotated around detected objects and saved locally as `.png` files 
- Wrong format error is displayed to the user in the terminal

## Key Features

- Multi-Image Support: Process multiple S3 object keys in one execution
- Automated Object Detection: Zero configuration Rekognition inference
- Visual Annotation: Bounding boxes makes clear the element being highlighted
- Terminal Reporting: Real Time display of detected labels and confidence metrics

## Requirements

To run this project locally, you will need:

- Python 3 and above
- AWS CLI configured with appropriate IAM credentials
- Python dependencies: 
    - boto3
    - Pillow
    - Matplotlib

## Permissions Required

- `s3:GetObject` for the source bucket
- `rekognition: DetectLabels`
- Local write permissions for output folder

## Setup and Installation

- Clone the repository: 

```bash
  git clone https://github.com/SohailMohammed1/AWS-Image-Label-Generator.git
  cd image-label-generator
```
- Create and activate a virtual environment:

```bash
    python -m venv env
    source env/bin/activate 
```
- Install dependencies:

```bash
    pip install -r requirements.txt
```
- Ensure your AWS credentials are configured:

```bash
    aws configure
```
## Usage

Run the application and pass one or more S3 key images:

```bash
python ImageLabelGenerator.py image1.jpg image2.webp image3.png
```

The programme will:
- Retrieve each image from S3
- Analyse it with Rekognition
- Produce annotated `.png` outputs
- Print the label summaries to the terminal

Example:

```bash
Detected labels for image1.jpg
Label: Car – Confidence: 98.34
Label: Person – Confidence: 92.10
Label: Road – Confidence: 88.02
```

## Error Handling

- The application incorporates handling for:
- Unsupported image formats (e.g., certain webp variants)
- Missing or inaccessible S3 objects
- Incorrect credentials or insufficient IAM permissions
- Images with no detectable objects
- Clear log messages guide the user towards resolution.


## Limitations

- Rekognition returns bounding boxes in relative coordinates; very large or cropped images may require manual scaling adjustments.
- Complex images with overlapping objects may generate dense bounding boxes.
- Certain formats (like animated .webp) may not render consistently through Pillow.

## Future Enhancements

- To demonstrate continued progression and cloud-native thinking, future improvements may include:
- Deploying as a Lambda-based serverless API
- Adding a simple UI or web dashboard
- Integrating DynamoDB for storing historical detection results
- Sending detection summaries to SNS or EventBridge
- Batch analysis through S3 event triggers
- Adding unit tests and CI/CD pipelines (GitHub Actions)

## Project Learnings and Takeaways

This project strengthened practical understanding of:

- AWS SDK (boto3) integrations
- Rekognition computer vision workflows
- S3 object management
- Image manipulation and annotation in Python
- Designing CLI-based automation tools
- Handling cloud permissions, errors, and multi-service interactions