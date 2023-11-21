import time
from typing import Dict

import boto3
import pandas as pd

import os 
from dotenv import load_dotenv
load_dotenv()
from io import StringIO

from langchain.llms.bedrock import Bedrock


SCHEMA_NAME = os.environ.get("SCHEMA_NAME").strip('"')
S3_STAGING_DIR = os.environ.get("S3_STAGING_DIR").strip('"')
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME").strip('"')
AWS_REGION = os.environ.get("AWS_REGION").strip('"')

# def create_aws_session():
#     Session = boto3.Session(
#         aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID").strip('"'),
#         aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY").strip('"'),
#         aws_session_token=os.environ.get("AWS_SESSION_TOKEN").strip('"'),
#         region_name=os.environ.get("REGION_NAME").strip('"')
#     )
#     return Session



def create_llm_anthropic():
    bedrock = boto3.client(
        service_name='bedrock',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID").strip('"'), 
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY").strip('"'), 
        region_name='us-east-1',
        endpoint_url='https://bedrock.us-east-1.amazonaws.com'
    )
    llm = Bedrock(model_id='anthropic.claude-v2', region_name='us-east-1', client=bedrock, model_kwargs={"max_tokens_to_sample": 5000})
    return llm

def create_athena_client():
    return  boto3.client('athena', 
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID").strip('"'), 
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY").strip('"'), 
                      region_name=os.environ.get("REGION_NAME").strip('"')
                      )


def create_s3_client():
    return  boto3.client('s3', 
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID").strip('"'), 
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY").strip('"'), 
                      region_name=os.environ.get("REGION_NAME").strip('"')
                      )


def load_query_results(
    client: boto3.client, query_response: Dict
) -> pd.DataFrame:
    while True:
        try:
            # This function only loads the first 1000 rows
            client.get_query_results(
                QueryExecutionId=query_response["QueryExecutionId"]
            )
            break
        except Exception as err:
            if "not yet finished" in str(err):
                time.sleep(0.001)
            else:
                raise err
    s3_client = create_s3_client()
    S3_OUTPUT_DIRECTORY = os.environ.get("S3_OUTPUT_DIRECTORY").strip('"')

    s3_object_path = f"{S3_OUTPUT_DIRECTORY}/{query_response['QueryExecutionId']}.csv"

    s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_object_path)
    data = s3_object['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    df.index = [''] * len(df)
    return df

