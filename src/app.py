import boto3
import os
from botocore.exceptions import ClientError

# Create clients
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')

# Load environment variables, provide defaults:
prompt_key_name = os.environ.get('PROMPT_KEY_NAME', 'prompt.txt')
bucket_name = os.environ.get('BUCKET_NAME', 'kk-app-bucket-1022')
model_id = os.environ.get('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')


# This function calls a Bedrock model using the given prompt + text:
def call_model(prompt, text):

    # Append the text to evaluate to the prompt:
    conversation = [{
            "role": "user",
            "content": [{"text": prompt + " " + text}],
        } ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = bedrock.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 1024, "temperature": 0.5, "topP": 0.9},
        )

        # Extract and return the response text.
        return response["output"]["message"]["content"][0]["text"]

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


# This function reads the given S3 bucket/object key into a String:
def read_s3_object(bucket_name, object_key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        object_data = response['Body'].read()
        object_string = object_data.decode('utf-8')
        return object_string

    except Exception as e:
        print(f"Error reading S3 object (Bucket: {bucket_name}, Key: {object_key}): {e}")
        raise

# Extract the bucket name and object key from an S3 event record.
def get_bucket_and_key(record):
    bucket_name = record['s3']['bucket']['name']
    object_key = record['s3']['object']['key']
    return bucket_name, object_key

# Process an individual S3 object by retrieving it and calling the AI model.
def process_s3_event(bucket_name, object_key):

    # ignore cases where the prompt.txt file is uploaded.
    # (For simplicity we keep the prompt file in the same bucket as the files to be processed):
    if object_key == prompt_key_name:
        print("Prompt file uploaded, ignoring.")
        return
    prompt = read_s3_object(bucket_name, prompt_key_name)

    print(f"Processing object: Bucket: {bucket_name}, Key: {object_key}")
    text = read_s3_object(bucket_name, object_key)
    response = call_model(prompt, text)
    print(f"Response: {response}")

# Entry point when called from a Lambda function.
# The expected event is an S3 object creation event:
def lambda_handler(event, context):
    for record in event['Records']:
        bucket_name, object_key = get_bucket_and_key(record)
        process_s3_event(bucket_name, object_key)
    print("Complete!")

# Entry point when the code is run from a command line:
def main():
    # Obtain a list of all objects in the bucket:
    response = s3.list_objects_v2(Bucket=bucket_name) 
    # Loop through each object in the response:
    for obj in response.get('Contents', []):
        object_key = obj['Key']
        process_s3_event(bucket_name, object_key)
    print("Complete!")


if __name__ == "__main__":
    main()