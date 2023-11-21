import json
import boto3
import os
import sqlalchemy
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from langchain.chains import SQLDatabaseChain
from langchain import PromptTemplate, SQLDatabase, SQLDatabaseChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import SQLDatabaseSequentialChain
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatAnthropic
import time
from langchain.llms.bedrock import Bedrock
from dotenv import load_dotenv
load_dotenv()
from pyathena import connect
from CustomSQLDatabaseChain import SQLDatabaseChain_Custom

from common_functions import create_aws_session, create_llm_anthropic

database_name = "ca_data"
s3_staging_dir = 's3://our-bedrock-multidata/multi_stage'

database_name = os.environ.get("SCHEMA_NAME").strip('"')
s3_staging_dir = os.environ.get("S3_STAGING_DIR").strip('"')


def create_db_connection(database_name, s3_staging_dir):
    AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY").strip('"')
    AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY").strip('"')
    SCHEMA_NAME = database_name
    S3_STAGING_DIR = s3_staging_dir
    AWS_REGION = os.environ.get("REGION_NAME").strip('"')
    ATHENA_WORK_GROUP = "primary"
    conn_str = (
        "awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}@"
        "athena.{region_name}.amazonaws.com:443/"
        "{schema_name}?s3_staging_dir={s3_staging_dir}&work_group={work_group}"
    )

    engine = create_engine(
        conn_str.format(
            aws_access_key_id=quote_plus(AWS_ACCESS_KEY),
            aws_secret_access_key=quote_plus(AWS_SECRET_KEY),
            region_name=AWS_REGION,
            schema_name=SCHEMA_NAME,
            s3_staging_dir=quote_plus(S3_STAGING_DIR),
            work_group=quote_plus(ATHENA_WORK_GROUP)
        )
    )
    athena_connection = engine.connect()

    dbathena = SQLDatabase(engine)
    db = dbathena
    return db



def run_query(query):
    DEFAULT_TEMPLATE = [
        {
            "role": "Human",
            "content": """
            Given an input question, first create a syntactically correct {dialect} query to run, then look at the SQL and return the answer as per the SQLResult only at the end.

            Do not append 'Query:' to SQLQuery. Only write the runnable SQLQuery in this line.

            Display SQLResult after the query is run in plain English that users can understand.

            Use approx_percentile instead of percentile_approx to calculate median.

            Use average to calculate mean.

            Display the answer in plain English.

            If you stop generating text in the middle, continue immediately.

            Only use the following tables:

            {table_info}


            Human: {input}
            """
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    formatted_prompt = "\n".join([f"{turn['role']}:{turn['content']}" for turn in DEFAULT_TEMPLATE])
    PROMPT_sql = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=formatted_prompt
    )

    db = create_db_connection(database_name, s3_staging_dir)
    llm = create_llm_anthropic()

    db_chain = SQLDatabaseChain_Custom.from_llm(llm, db, prompt=PROMPT_sql, verbose=True, return_intermediate_steps=False, top_k=1000, use_query_checker=False, return_direct=False)
    response = db_chain.run(query)

    return response

if __name__ == "__main__":
    pass
