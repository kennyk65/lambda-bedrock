# Lambda Bedrock

This SAM-based application demonstrates how to work with S3, Lambda, and Bedrock-hosted models to perform some basic generative AI tasks.

When a file is uploaded to the specified bucket, the Lambda function will read it, append it to the end of a flexible prompt, call an Amazon Bedrock model, and print the results to the log.
 
## Requirements
* Install SAM local
* AWS Account

## Usage
* Within your AWS account, go to Amazon Bedrock and __enable__ the model(s) you wish to use.  For this demo I've enabled the _anthropic.claude-3-sonnet-20240229-v1:0_ model in us-west-2, but you can play with any other text models enabled in any region.
* Install the application with `sam deploy`.  You will probably need to override the bucket name I've chosen, and perhaps the model if you've chosen something different.
* Upload a `prompt.txt` file to the newly created S3 bucket.  Here is a suggestion:

```
Please evaluate the correctness of the following assessment of the weather.  Use the following examples to guide your analysis.  Your response should be a SINGLE WORD indicating if the assessment is correct or incorrect.

EXAMPLES:

REPORT:  The current weather is 78 degrees and mostly sunny
ASSESSMENT:  Pleasant
ACCURACY OF ASSESSMENT:  Correct

REPORT:  A winter storm advisory is in effect, motorists are urged not to travel
ASSESSMENT: Harsh, Dangerous
ACCURACY OF ASSESSMENT:  Correct

REPORT:  A fresh layer of snow has blanketed the resort, which should permit powder skiing.
ASSESSMENT: Unpleasant
ACCURACY OF ASSESSMENT:  Incorrect


Analyze the following:

```

* Upload a test file to the S3 bucket, something like:

```
REPORT:  Heavy fog will be encountered early, followed by heavy ran and possible hail.
ASSESSMENT:  foul
```
  Open CloudWatch logs to find the log output.

* Test the application locally (if you like) with `sam local invoke --event events/event.json`. (The test assumes you have deployed the application to install the S3 bucket).  Make sure the name of the S3 object defined in event.json matches the file you just uploaded.

* Test the application on AWS by uploading other test files modeled like the example above. 

---


This project contains source code and supporting files for a serverless application that you can deploy with the SAM CLI. It includes the following files and folders.

- hello_world - Python Code for the application's Lambda function.
- events - Invocation events that you can use to invoke the function, such as `sam local invoke --event events/event.json`.
- tests - Unit tests for the application code. 
- template.yaml - A template that defines the application's AWS resources.

The application uses several AWS resources, including Lambda functions and an API Gateway API. These resources are defined in the `template.yaml` file in this project. You can update the template to add AWS resources through the same deployment process that updates your application code.

If you prefer to use an integrated development environment (IDE) to build and test your application, you can use the AWS Toolkit.  
The AWS Toolkit is an open source plug-in for popular IDEs that uses the SAM CLI to build and deploy serverless applications on AWS. The AWS Toolkit also adds a simplified step-through debugging experience for Lambda function code. See the following links to get started.

* [CLion](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [GoLand](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [IntelliJ](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [WebStorm](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [Rider](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [PhpStorm](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [PyCharm](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [RubyMine](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [DataGrip](https://docs.aws.amazon.com/toolkit-for-jetbrains/latest/userguide/welcome.html)
* [VS Code](https://docs.aws.amazon.com/toolkit-for-vscode/latest/userguide/welcome.html)
* [Visual Studio](https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/welcome.html)

## Deploy the sample application

The Serverless Application Model Command Line Interface (SAM CLI) is an extension of the AWS CLI that adds functionality for building and testing Lambda applications. It uses Docker to run your functions in an Amazon Linux environment that matches Lambda. It can also emulate your application's build environment and API.

To use the SAM CLI, you need the following tools.

* SAM CLI - [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
* [Python 3 installed](https://www.python.org/downloads/)
* Docker - [Install Docker community edition](https://hub.docker.com/search/?type=edition&offering=community)

* **Stack Name**: The name of the stack to deploy to CloudFormation. This should be unique to your account and region, and a good starting point would be something matching your project name.
* **AWS Region**: The AWS region you want to deploy your app to.
* **Confirm changes before deploy**: If set to yes, any change sets will be shown to you before execution for manual review. If set to no, the AWS SAM CLI will automatically deploy application changes.
* **Allow SAM CLI IAM role creation**: Many AWS SAM templates, including this example, create AWS IAM roles required for the AWS Lambda function(s) included to access AWS services. By default, these are scoped down to minimum required permissions. To deploy an AWS CloudFormation stack which creates or modifies IAM roles, the `CAPABILITY_IAM` value for `capabilities` must be provided. If permission isn't provided through this prompt, to deploy this example you must explicitly pass `--capabilities CAPABILITY_IAM` to the `sam deploy` command.
* **Save arguments to samconfig.toml**: If set to yes, your choices will be saved to a configuration file inside the project, so that in the future you can just re-run `sam deploy` without parameters to deploy changes to your application.

You can find your API Gateway Endpoint URL in the output values displayed after deployment.

## Use the SAM CLI to build and test locally

Build your application with the `sam build --use-container` command.

```bash
sam build --use-container
```

The SAM CLI installs dependencies defined in `hello_world/requirements.txt`, creates a deployment package, and saves it in the `.aws-sam/build` folder.

Test a single function by invoking it directly with a test event. An event is a JSON document that represents the input that the function receives from the event source. Test events are included in the `events` folder in this project.

Run functions locally and invoke them with the `sam local invoke` command.

```bash
sam-app-python$ sam local invoke --event events/event.json
```


## Cleanup

The SAM application generates costs only when it is in use; if no files are being processed there are no charges.  To delete the sample application that you created, use the AWS CLI. Assuming you used your project name for the stack name, you can run the following:

```bash
sam delete 
```

## Resources

See the [AWS SAM developer guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html) for an introduction to SAM specification, the SAM CLI, and serverless application concepts.

Next, you can use AWS Serverless Application Repository to deploy ready to use Apps that go beyond hello world samples and learn how authors developed their applications: [AWS Serverless Application Repository main page](https://aws.amazon.com/serverless/serverlessrepo/)
