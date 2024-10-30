import json
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.store.memory import InMemoryStore
from langchain.chat_models.base import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SQLDatabase

from src.constant import DB_PATH
from src.state import AgentState

class SQLAgentRAG:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Any,
        db_uri: str = DB_PATH,
        table_json_path: str = "../backend/data/table.jsonl",
        column_json_path: str = "../backend/data/column.jsonl",
        
    ):
        self.llm = llm
        self.table_json_path = table_json_path
        self.column_json_path = column_json_path
        self.db = SQLDatabase.from_uri(db_uri)
        self.schema = None
        self.retriever = tools

         # add nodes
        graph = StateGraph(AgentState)
        graph.add_node("router", self.router_node)
        graph.add_node("general_asistant", self._general_asistant)
        graph.add_node("sql_gen", self._sql_gen)
        graph.add_node("validate_sql", self._validate_sql)
        graph.add_node("solve_error", self._solve_error)
        graph.add_node("response", self._query_gen_node)

        # add edges
        graph.set_entry_point("router")
        # graph.add_edge(START, "sql_gen")
        graph.add_edge("sql_gen", "validate_sql")
        graph.add_conditional_edges(
            "router",
            self.router,
            {
                "SQL": "sql_gen",
                "GENERAL": "general_asistant"
            }
        )
        graph.add_conditional_edges(
            "validate_sql",
            self._should_continue
        )
        graph.add_edge("solve_error", "validate_sql")
        graph.add_edge("response", END)
        graph.add_edge("general_asistant", END)

        # compile
        store = InMemoryStore()
        checkpointer = MemorySaver()
        self.graph = graph.compile(checkpointer=checkpointer, store=store)


    def _indexing_table(self, query: str):
        """
        Index and retrieve relevant tables based on the input query.
        """
        docs_table = JSONLoader(
            file_path=self.table_json_path,
            jq_schema='.',
            text_content=False,
            json_lines=True
        ).load()

        retriever = self.retriever(docs_table, k=5, search_type='mmr', lambda_mult=1)

        matched_documents_table = retriever.invoke(query)
        matched_tables = [
            json.loads(doc.page_content)["table"] for doc in matched_documents_table
        ]

        return matched_tables
    
    def _indexing_column(self, matched_tables, query: str):
        """
        Index and retrieve relevant columns based on the matched tables and query.
        """
        docs_column = JSONLoader(
            file_path=self.column_json_path,
            jq_schema='.',
            text_content=False,
            json_lines=True
        ).load()

        retriever = self.retriever(docs_column, k=20, search_type='similarity')

        matched_columns = retriever.invoke(query)
        matched_columns_filtered = [
            json.loads(doc.page_content) for doc in matched_columns
            if json.loads(doc.page_content)["table_name"] in matched_tables
        ]

        matched_columns_cleaned = [
            f'table_name={doc["table_name"]}|column_name={doc["column_name"]}|data_type={doc["data_type"]}'
            for doc in matched_columns_filtered
        ]

        return matched_columns_cleaned
    
    def _sql_gen(self, state: AgentState):
        """
        Generates a SQL query based on the input provided by the user.
        This function uses the LLM to construct the query from matched tables and columns.
        """
        messages = state["messages"][-1].content
        matched_table = self._indexing_table(messages)
        self.schema = self._indexing_column(matched_table, messages)

        prompt = PromptTemplate(
            template="""
              You are a SQL master expert specializing in writing complex SQL queries for SQLite. Your task is to construct a SQL query based on the provided information. Follow these strict rules:

              QUERY: {query}
              -------
              MATCHED_SCHEMA: {matched_schema}
              -------

              Please construct a SQL query using the MATCHED_SCHEMA and the QUERY provided above.
              The goal is to determine the availability of hotels based on the provided info.

              IMPORTANT: Use ONLY the column names (column_name) mentioned in MATCHED_SCHEMA. DO NOT USE any other column names outside of this.
              IMPORTANT: Associate column_name mentioned in MATCHED_SCHEMA only to the table_name specified under MATCHED_SCHEMA.
              NOTE: Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed.
              Generate ONLY the SQL query. Do not provide any explanations, comments, or additional text.

            """,
            input_variables=["query", "matched_schema"]
        )

        sql_gen = prompt | self.llm | StrOutputParser()
        result_sql = sql_gen.invoke({"query": messages, "matched_schema": self.schema})
        return {"sql_query": [result_sql]}
    
    def _validate_sql(self, state: AgentState):
        """
        Validates the generated SQL query by attempting to execute it.
        Returns success if no errors, otherwise returns the error message.
        """
        query = state["sql_query"][-1]
        try:
            result = self.db.run(query)

        except Exception as e:
            return {"error_str": [f"Unexpected Error: {str(e)}"]}
        
    def _solve_error(self, state: AgentState):
        """
        Called with the error code and error description as the argument to get guidance on how to solve the error
        """
        error_string = state["error_str"][-1]
        sql_query = state["sql_query"][-1]
        prompt = PromptTemplate(
            template="""
              First, identify the main issues with the given SQL query based on the error message.
              {error_string}

              Next, examine the schema and current SQL query to locate potential sources of the error.
              {schema}

              Then, modify the current SQL query to fix the error and avoid similar issues in the future.
              {sql_query}

              Finally, ensure the revised SQL query conforms to the requirements outlined in the original task and provide the corrected SQL query.
              Generate ONLY the SQL query. Do not provide any explanations, comments, or additional text.

            """,
            input_variables=["error_string", "schema", "sql_query"]
        )
        resolver = prompt | self.llm | StrOutputParser()
        response = resolver.invoke({"error_string": error_string, "schema": self.schema, "sql_query": sql_query})
        return {"sql_query": [response]}
    
    def _query_gen_node(self, state: AgentState):
        """
        Generates a final response after executing the SQL query and getting the result.
        """
        query = state["sql_query"][-1]
        messages = state["messages"][-1]  
        prompt = PromptTemplate(
            template="""Based on the following SQL result, generate a natural language response:
            Query SQL: {user_query}
            SQL Result: {sql_response}
            """,
            input_variables=["user_query", "sql_response"]
        )

        sql_response = self.db.run(query)
        response_llm = prompt | self.llm | StrOutputParser()
        response = response_llm.invoke({"user_query": messages, "sql_response": sql_response})
        return {"messages": [response]}
    
    def _general_asistant(self, state: AgentState):
        messages = state["messages"]
        response = self.llm.invoke(messages)   # Pertama berhasil

        return {'messages': [response]}

    def router(self, state: AgentState):
        return state["question_type"]

    def router_node(self, state: AgentState):
        question = state["messages"][-1].content
        prompt = PromptTemplate(
            template= """
            You are a senior specialist of analytical support. Classify incoming questions into one of two types:
              - SQL: Related to flight information, schedules, hotels, rentals, recommendations, and anything about vacations
              - GENERAL: General questions
            Return only one word: SQL, or GENERAL.

            {question}
        """,
            input_variables=["question"]
        )
        router = prompt | self.llm | StrOutputParser()
        question_type = router.invoke({"question": question})
        return {"question_type": question_type}

    def _should_continue(self, state: AgentState) -> Literal["response", "solve_error"]:
        """
        Decides whether to proceed based on SQL validation results.
        If the last message contains "correct", continue to response.
        Otherwise, go to error-solving or retry.
        """
        if "error_str" in state:
            # If 'error_str' exists and contains an error, return "solve_error"
            return "solve_error"
        else:
            return "response"  # If no error, proceed to generate the final response
