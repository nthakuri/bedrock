"""Chain for interacting with SQL Database."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra, Field
import os
import re
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import  PROMPT, SQL_PROMPTS
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase

import common_functions

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


class SQLDatabaseChain_Custom(Chain):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python
 
            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SQLDatabaseChain.from_llm(OpenAI(), db)
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    database: SQLDatabase = Field(exclude=True)
    """SQL Database to connect to."""
    prompt: Optional[BasePromptTemplate] = None
    """[Deprecated] Prompt to use to translate natural language to SQL."""
    top_k: int = 8
    """Number of results to return from the query"""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""
    use_query_checker: bool = False
    """Whether or not the query checker tool should be used to attempt 
    to fix the initial SQL from the LLM."""
    query_checker_prompt: Optional[BasePromptTemplate] = None
    """The prompt template that should be used by the query checker"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True


    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, INTERMEDIATE_STEPS_KEY]

    def _call(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": f'You are the helpful assistant which generates SQL Query for given {input_text}. You are forbidden to write any thing else other than SQL Query always add ";" semicolon at the end of sql query.',
            "top_k": str(self.top_k),
            "dialect": self.database.dialect,
            "table_info": table_info,
            "temperature" : 0,
            "maxTokenCount": -1,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps_to_display: List = []

        try:
            sql_cmd = self.llm_chain.predict(
                **llm_inputs,
            ).strip()


            sql_cmd = re.findall(r'\bSELECT\b.*?;', sql_cmd, re.IGNORECASE | re.DOTALL)[0]
            athena_client = common_functions.create_athena_client()
            SCHEMA_NAME = os.environ.get("SCHEMA_NAME").strip('"')
            S3_STAGING_DIR = os.environ.get("S3_STAGING_DIR").strip('"')

            if not self.use_query_checker:
                intermediate_steps_to_display.append(f'SQL :{sql_cmd}')
                try:
                    response = athena_client.start_query_execution(
                        QueryString=sql_cmd,
                        QueryExecutionContext={"Database": SCHEMA_NAME},
                        ResultConfiguration={
                            "OutputLocation": S3_STAGING_DIR,
                            "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
                        },
                    )
                except athena_client.exceptions.InvalidRequestException as e:
                    result = "Could not run Query"
                else:    
                    result = common_functions.load_query_results(athena_client, response)

                if result[1] == True:
                    returned_data = len(result[0])
                    intermediate_steps_to_display.append(f"The query returned {returned_data} rows")
                    intermediate_steps_to_display.append(f'Result:\n{result[0]}')
                if result[1] == False:
                    self.return_direct = True

            if self.return_direct:
                final_result = result[0]
            else:
                input_text += f"""\nYou are the helpful Assistant which replies to above question taking references through\n{sql_cmd}\n
                SQLResult: {result}\n

                You will get SQL Result in the form of :
                
                (column_1)             (column_2)......
                column_1 row 0 data    column_2 row 0 data .........
                column_1 row 1 data    column_2 row 1 data .........
                ...          ...                    ...                 ...

                Only look at the data below the column name like column_1, column_2 etc...Note: reply with the complete answer from SQLResult do not leave any result.

                Only reply the answer that human asked\n 
                take this as example:for the question :
                How many rows are in the table? Answer: There are 1234 rows in the table. where 1234 is content from the sql result.\n 
                For the question: What are the top ten billing codes in descending order?
                Answer:The top ten billing codes in descending order are:
                None, 90834, 95806, 99213, 99406, 99423, 99422, 99407, 99214, and 99215.\n
                """

                llm_inputs["input"] = input_text
                # intermediate_steps.append(llm_inputs)  # input: final answer
                

                try:
                    flag = 0
                    # Attempt to call the Bedrock model with your original prompt
                    final_result = self.llm_chain.predict(
                        # callbacks=_run_manager.get_child(),
                        **llm_inputs,
                    ).strip()


                except Exception as e:
                    if "ValidationException" in str(e):
                        # Handle the ValidationException by performing another prompt
                        final_result = "Your output data is too long"
                    else:
                        # Handle other exceptions as needed
                        print(f"An error occurred: {e}")



                intermediate_steps_to_display.append(f'Final Result:\n{final_result}')


            chain_result: Dict[str, Any] = {self.output_key: str(intermediate_steps_to_display)}


            return  chain_result
        


        except Exception as exc:

            exc.intermediate_steps = intermediate_steps_to_display # type: ignore
            raise exc

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        db: SQLDatabase,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> SQLDatabaseChain_Custom:
        prompt = prompt or SQL_PROMPTS.get(db.dialect, PROMPT)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, database=db, **kwargs)