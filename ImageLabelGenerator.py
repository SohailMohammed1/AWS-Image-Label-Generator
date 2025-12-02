import argparse 
import boto3
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

def generate_labels_for_image(source: str):

    def detect_labels(photo, bucket):
        client = boto3.client('rekognition')

    response = client.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        MaxLabels=10
    )

    print('Detected labels for ' + photo)
    print()
    for label in response['Labels']:
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

    # 🔽 Save instead of show
    output_filename = "rekognition_output.png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

    print(f"\nSaved output image with bounding boxes to: {output_filename}")

    return len(response['Labels'])

def main():
    photo = 'istockphoto-1029925066-612x612.jpg'
    bucket = 'sohailm2-aws-rekognition-label-images'
    label_count = detect_labels(photo, bucket)
    print("Labels detected:", label_count)

if __name__ == "__main__":
    main()


def main():
    parser = argparse.ArgumentParser(
        description="Image Label Generator - analyze a single image"
    )

    # One positional argument: the image source
    parser.add_argument(
        "image",
        help="Image URL or file path to analyze",
    )

    args = parser.parse_args()

    image_source = args.image

    # Call your analysis function
    labels = generate_labels_for_image(image_source)

    # Nicely print the result
    print(f"\nImage: {image_source}")
    if labels:
        print("Labels:")
        for label in labels:
            print(f"  - {label}")
    else:
        print("No labels found.")


if __name__ == "__main__":
    main()














# import boto3
# import matplotlib
# matplotlib.use("Agg")  
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# from io import BytesIO


# def detect_labels(photo, bucket):
#     client = boto3.client('rekognition')

#     response = client.detect_labels(
#         Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
#         MaxLabels=10
#     )

#     print('Detected labels for ' + photo)
#     print()
#     for label in response['Labels']:
#         print("Label:", label['Name'])
#         print("Confidence:", label['Confidence'])
#         print()

#     # Load image from S3
#     s3 = boto3.resource('s3')
#     obj = s3.Object(bucket, photo)
#     img_data = obj.get()['Body'].read()
#     img = Image.open(BytesIO(img_data))

#     # Plot image with bounding boxes
#     plt.figure(figsize=(8, 6))
#     plt.imshow(img)
#     ax = plt.gca()

#     for label in response['Labels']:
#         for instance in label.get('Instances', []):
#             bbox = instance['BoundingBox']
#             left = bbox['Left'] * img.width
#             top = bbox['Top'] * img.height
#             width = bbox['Width'] * img.width
#             height = bbox['Height'] * img.height

#             rect = patches.Rectangle(
#                 (left, top),
#                 width,
#                 height,
#                 linewidth=1,
#                 edgecolor='r',
#                 facecolor='none'
#             )
#             ax.add_patch(rect)

#             label_text = f"{label['Name']} ({round(label['Confidence'], 2)}%)"
#             plt.text(
#                 left,
#                 top - 2,
#                 label_text,
#                 color='r',
#                 fontsize=8,
#                 bbox=dict(facecolor='white', alpha=0.7)
#             )

#     plt.axis('off')

#     # 🔽 Save instead of show
#     output_filename = "rekognition_output.png"
#     plt.savefig(output_filename, bbox_inches='tight')
#     plt.close()

#     print(f"\nSaved output image with bounding boxes to: {output_filename}")

#     return len(response['Labels'])

# def main():
#     photo = 'istockphoto-1029925066-612x612.jpg'
#     bucket = 'sohailm2-aws-rekognition-label-images'
#     label_count = detect_labels(photo, bucket)
#     print("Labels detected:", label_count)

# if __name__ == "__main__":
#     main()
