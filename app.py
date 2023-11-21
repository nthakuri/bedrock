import streamlit as st
import ast
from bedrock import run_query  # Import the run_query function from core.py

def main():
    st.title("SQL Query and Answer Generator")

    user_input = st.text_input("Enter your SQL query:")
    run_query_button = st.button("Run Query")  # Move the button creation here

    if run_query_button:
        response = run_query(user_input)
        input_list = ast.literal_eval(response)

        for item in input_list:
            st.text(item)

if __name__ == "__main__":
    main()
