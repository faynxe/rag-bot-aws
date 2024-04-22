## My Project

This project walks you through how yoou can deploy a ChatBot that uses the Retrieval Augmented Generation (RAG) Technique. This ChatBot is Powered by [Amazon Bedrock](https://aws.amazon.com/bedrock/)/[Amazon SageMaker JumStart](https://aws.amazon.com/sagemaker/jumpstart/) LLM/Embedding Models and uses [Amazon Kendra](https://aws.amazon.com/kendra/) and/or [Amazon OpenSearch](https://aws.amazon.com/opensearch-service/) as Vector Storage and Retrieval Engines and or Bedrock Knowledge Base.

## Git Contents
```
-Images
-Prompt_Template 
    |------- Rag (Contains prompt template for Retrieval Augmented Generation (RAG))
    |------- Summary (Contains prompt template for Summarization)
- application_metadata_complete.json (app configuration file)
- config.json
- rag-bot.py (streamlit app)
- requirements.txt
- StreamlitLink.ipynb
```

## PreRequisite
You would need to set up the following before succesfully using this app:
1. [Create Kendra Index](https://docs.aws.amazon.com/kendra/latest/dg/create-index.html) AND/OR
2. [Create OpenSearch Domain](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/createupdatedomains.html)
   - For the opensearch domain, Enable `fine-grained access control` and **Set IAM ARN as master user**.
   - For Access policy, select **Only use fine-grained access control**
3. [Create a Bedrock Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html)
3. [Deploy HuggingFace Models with Amazon SageMaker JumpStart](https://www.philschmid.de/sagemaker-llama-llm) (Optional)
4. Amazon Bedrock Access.
5. Deploy HuggingFace Embedding Models with SageMaker JumpStart (Optional)
6. [Create an Amazon DyanmoBD Table with UserID as primary Key](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/getting-started-step-1.html) (Optional)
7. Modify the application_metadata_complete.json file with the correct values of your AWS account:
    - llama2-7b value of sagemaker endpoint name **    
    - embedding model values of sagemaker endpoint name (other than cohere and titan keys) **
    - HuggingFace api token for hugginfacekey value **
    - Kendra bucket and prefix values where uploaded file would be stored in Amazon S3
    - Kendra execution role value
    - Kendra index value 
    - Kendra S3 data source name
    - Amazon OpenSearch parameters (domain endpoint **e.g. my-test-domain.us-east-1.es.amazonaws.com**)    
    ** Optional 
    
## Configuration
The application requires several configuration files to be set up correctly. Follow these steps to modify the configuration files:

1. application_metadata_complete.json
This file contains metadata about the AWS services and resources used by the application. Update the following fields according to your AWS environment:

models-llm: List of language models and their endpoints.
models-emb: List of embedding models and their endpoints.
region: The AWS region where your resources are located.
Kendra: Configuration for Amazon Kendra, including the bucket name, prefix, index ID, and role ARN.
opensearch: Configuration for OpenSearch, including the domain endpoint and domain name.
KnowledgeBase: Configuration for KnowledgeBase, including the index ID and data source ID.
2. config.json
This file contains additional configuration settings for the application:

DynamodbTable: The name of the DynamoDB table used for storing chat history.
UserId: The user ID associated with the chat history.
Bucket_Name: The name of the S3 bucket used for storing documents.
max-output-token: The maximum number of output tokens allowed for the language model.
cognito-app-id: The Cognito app ID (if using Cognito authentication).
use-cognito: A boolean flag indicating whether to use Cognito authentication.
chat-history-loaded-length: The number of chat history entries to load.
bedrock-region: The AWS region where the Bedrock service is located.
load-doc-in-chat-history: A boolean flag indicating whether to load documents in the chat history.
    
   
## Set Up StreamLit Front-End
The streamlit app for this prioject is located in *rag_advanced.py*.

To run this Streamlit App on Sagemaker Studio follow the steps in the link below:
* [Set Up SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)
* SageMaker execution role should have access to interact with [SageMaker Runtime](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html), [Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/api-setup.html), [OpenSearch](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/fgac-http-auth.html), [ComprehendReadOnly Access](https://docs.aws.amazon.com/comprehend/latest/dg/security-iam-awsmanpol.html) , [Textract](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonTextractFullAccess.html), [DynamoDB] (https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/iam-policy-specific-table-indexes.html), [Kendra](https://docs.aws.amazon.com/kendra/latest/dg/security-iam-awsmanpol.html) and [S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-policy-language-overview.html).
* [Launch SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-launch.html)
* [Clone this git repo into studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tasks-git.html)
* Open a system terminal by clicking on **Amazon SageMaker Studio** and then **System Terminal** as shown in the diagram below
* <img src="images/studio-new-launcher.png" width="600"/>
* Navigate into the cloned repository directory using the `cd` command and run the command `pip install -r requirements.txt` to install the needed python libraries
* Run command `python3 -m streamlit run rag-bot.py --server.enableXsrfProtection false --server.enableCORS  false` to start the Streamlit server. Do not use the links generated by the command as they won't work in studio.
* To enter the Streamlit app, open and run the cell in the **StreamlitLink.ipynb** notebook. This will generate the appropiate link to enter your Streamlit app from SageMaker studio. Click on the link to enter your Streamlit app.
* **⚠ Note:**  If you rerun the Streamlit server it may use a different port. Take not of the port used (port number is the last 4 digit number after the last : (colon)) and modify the `port` variable in the `StreamlitLink.ipynb` notebook to get the correct link.

To run this Streamlit App on AWS EC2 (I tested this on the Ubuntu Image)
* [Create a new ec2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
* Expose TCP port range 8500-8510 on Inbound connections of the attached Security group to the ec2 instance. TCP port 8501 is needed for Streamlit to work. See image below
* <img src="images/sg-rules.PNG" width="600"/>
* [Connect to your ec2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html)
* Run the appropiate commands to update the ec2 instance (`sudo apt update` and `sudo apt upgrade` -for Ubuntu)
* Clone this git repo `git clone [github_link]`
* Install python3 and pip if not already installed
* EC2 [instance profile role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2_instance-profiles.html) has the required permissions to access the services used by this application mentioned above.
* Install the dependencies in the requirements.txt file by running the command `sudo pip install -r requirements.txt`
* Run command `python3 -m streamlit run rag-bot.py` 
* Copy the external link and paste in a new browser tab



## Note
Only Pdf documents are supported for OpenSearch upload at this time using this app.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

