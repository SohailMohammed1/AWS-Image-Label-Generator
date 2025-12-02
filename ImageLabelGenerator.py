import argparse 
import boto3
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO


def detect_labels(photo: str, bucket: str):
    """
    Call Rekognition on an S3 image, print labels + confidences,
    draw bounding boxes, save an output image, and return the label names.
    """
    client = boto3.client('rekognition')

    response = client.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        MaxLabels=10
    )

    print('Detected labels for ' + photo)
    print()

    label_names = []  # Collect label names to return

    for label in response['Labels']:
        label_names.append(label['Name'])
        print("Label:", label['Name'])
        print("Confidence:", label['Confidence'])
        print()

    # Load image from S3
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, photo)
    img_data = obj.get()['Body'].read()
    img = Image.open(BytesIO(img_data))

    # Plot image with bounding boxes
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    ax = plt.gca()

    for label in response['Labels']:
        for instance in label.get('Instances', []):
            bbox = instance['BoundingBox']
            left = bbox['Left'] * img.width
            top = bbox['Top'] * img.height
            width = bbox['Width'] * img.width
            height = bbox['Height'] * img.height

            rect = patches.Rectangle(
                (left, top),
                width,
                height,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

            label_text = f"{label['Name']} ({round(label['Confidence'], 2)}%)"
            plt.text(
                left,
                top - 2,
                label_text,
                color='r',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7)
            )

    plt.axis('off')

    # Save instead of show
    output_filename = "rekognition_output.png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

    print(f"\nSaved output image with bounding boxes to: {output_filename}")
    return label_names


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single S3 image with Amazon Rekognition"
    )

    # S3 object key 
    parser.add_argument(
        "photo",
        help="S3 object key (file name) to analyze, e.g. istockphoto-1029925066-612x612.jpg",
    )

    # Bucket
    parser.add_argument(
        "-b",
        "--bucket",
        default="sohailm2-aws-rekognition-label-images",
        help="S3 bucket name (default: sohailm2-aws-rekognition-label-images)",
    )

    args = parser.parse_args()

    photo = args.photo
    bucket = args.bucket

    labels = detect_labels(photo, bucket)

    # Summary
    print("\nSummary")
    print("-------")
    print(f"Image: s3://{bucket}/{photo}")
    print(f"Total labels detected: {len(labels)}")

    if labels:
        print("Labels:")
        for name in labels:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
