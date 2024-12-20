import boto3
import os
from botocore.exceptions import ClientError

# Create clients
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')

# Load environment variables, provide defaults:
prompt_key_name = os.environ.get('PROMPT_KEY_NAME', 'prompt.txt')
bucket_name = os.environ.get('BUCKET_NAME', 'my-bucket')
model_id = os.environ.get('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')

def call_model(prompt, text):

    # Append the text to evaluate to the prompt:
    user_message = prompt + " " + text 

    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

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


# define a function which will read the given bucket / key from S3 and return a string:
def read_s3_object(bucket_name, object_key):
    """
    Read an object from S3 and return its contents as a string.
    """
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        object_data = response['Body'].read()
        object_string = object_data.decode('utf-8')
        return object_string

    except Exception as e:
        print(f"Error reading S3 object (Bucket: {bucket_name}, Key: {object_key}): {e}")
        raise


def process_s3_event(record):
    """
    Process an individual S3 event record by retrieving the object from S3 and logging its contents.
    """
    bucket_name = record['s3']['bucket']['name']
    object_key = record['s3']['object']['key']

    # ignore cases where the prompt.txt file is uploaded:
    if object_key == prompt_key_name:
        print("Prompt file uploaded, ignoring.")
        return
    prompt = read_s3_object(bucket_name, prompt_key_name)

    print(f"Processing object: Bucket: {bucket_name}, Key: {object_key}")
    text = read_s3_object(bucket_name, object_key)
    response = call_model(prompt, text)
    print(f"Response: {response}")



def lambda_handler(event, context):
    """
    Lambda function handler that processes an event from S3 and logs object payloads.
    """
    for record in event['Records']:
        process_s3_event(record)

    print("Complete!")

# Example usage: event simulation (to be replaced by actual Lambda event in AWS environment)
# s3_json_logger_handler(event, context)
