from typing import Annotated, Literal, Optional
from langchain_core.messages import AIMessage, ToolMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class SQLAgent:
    """An agent that can answer questions about a SQL database by generating and executing queries.
    
    The agent follows these steps:
    1. Lists available tables
    2. Gets schema for relevant tables
    3. Generates SQL query
    4. Checks query for common mistakes
    5. Executes query and handles errors
    6. Formats final response
    """
    def __init__(self, db_uri: str, model: str = "gpt-4"):
        """Initialize SQLAgent with database connection and tools.
        
        Args:
            db_uri: Database connection string
            model: Name of the LLM model to use
            
        Raises:
            Exception: If database connection fails
        """
        try:
            # Test database connection
            self.db = SQLDatabase.from_uri(db_uri)
            # Try a simple query to verify connection
            self.db.run("SELECT 1")
        except Exception as e:
            raise Exception(f"Failed to connect to database: {str(e)}")
        
        self.model = model
        # Initialize tools
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=ChatOpenAI(model=model))
        self.tools = self.toolkit.get_tools()
        self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")
        self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
    
        # Setup the workflow
        self.app = self._create_workflow()

    @classmethod
    def handle_tool_error(state) -> dict:
        """Formats tool errors into user-friendly messages."""
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error executing query: {repr(error)}\nPlease revise the query and try again.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def _setup_query_generator(self):
        query_gen_system = """You are a SQL expert with a strong attention to detail.

        Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

        DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

        Before generating the query, consider the following:

        1. Carefully consider the relationships among all the tables base on its meaning and foreign keys. There could be multiple tables that are relevant to the question.
        2. Many of the tables has foreign keys to other tables, which means that the data in the table is related to the data in the other tables.
        3. There're many associations through intermediate tables, which means that the data in the table is related to the data in the other tables.
        4. Make sure you understand the meaning of all the columns and tables before generating the query.
        5. Most associations through tables has both source and target on its name, which means that the data in the table is related to the data in the other tables.
        6. Though there're cases that the association table doesn't have the source and target on its name, you should still consider the relationships among all the tables.

        When generating the query:

        Carefully consider the relationships among all the tables base on its meaning and foreign keys. There could be multiple tables that are relevant to the question.

        Output the SQL query that answers the input question without a tool call. Return only the SQL query, no other text.

        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.

        If you get an error while executing a query, rewrite the query and try again.

        If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
        NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

        If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
        
        query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", query_gen_system),
            ("placeholder", "{messages}")
        ])
        
        return query_gen_prompt | ChatOpenAI(model=self.model, temperature=0).bind_tools(
            [SubmitFinalAnswer]
        )

    def query_gen_node(self, state: State):
        message = self.query_gen.invoke(state)
        
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        return {"messages": [message] + tool_messages}

    def _create_workflow(self):
        # Create workflow
        workflow = StateGraph(State)
        
        def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sql_db_list_tables",
                                "args": {},
                                "id": "tool_abcd123",
                            }
                        ],
                    )
                ]
            }

        # Create tool nodes using instance method instead of classmethod
        def create_tool_node(tools):
            """Create a ToolNode with fallback handling."""
            return ToolNode(tools).with_fallbacks(
                [RunnableLambda(SQLAgent.handle_tool_error)], 
                exception_key="error"
            )

        # Add your model nodes
        model_get_schema = ChatOpenAI(model=self.model, temperature=0).bind_tools([self.get_schema_tool])

        @tool
        def db_query_tool(query: str) -> str:
            """Execute a SQL query against the database."""
            result = self.db.run_no_throw(query)
            if not result:
                return "Error: Query failed. Please rewrite your query and try again."
            return result

        def _setup_query_checker():
            query_check_system = """You are a SQL expert with a strong attention to detail.
            Double check the PostgresSQL query for common mistakes, including:
            - Using NOT IN with NULL values
            - Using UNION when UNION ALL should have been used
            - Using BETWEEN for exclusive ranges
            - Data type mismatch in predicates
            - Properly quoting identifiers
            - Using the correct number of arguments for functions
            - Casting to the correct data type
            - Using the proper columns for joins

            If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

            Reason through the query and the question and make sure the query is getting the answer you expect not contrary or mistaken its meaning.
            If the query is not getting the answer you expect, rewrite the query.

            You will call the appropriate tool to execute the query after running this check."""
            
            query_check_prompt = ChatPromptTemplate.from_messages([
                ("system", query_check_system),
                ("placeholder", "{messages}")
            ])
            
            return query_check_prompt | ChatOpenAI(model=self.model, temperature=0).bind_tools(
                [db_query_tool], tool_choice="required"
            )        

        # Add query check and generator
        self.query_check = _setup_query_checker()
        self.query_gen = self._setup_query_generator()

        def model_check_query(state: State) -> dict[str, list[AIMessage]]:
            """
            Use this tool to double-check if your query is correct before executing it.
            """
            return {"messages": [self.query_check.invoke({"messages": [state["messages"][-1]]})]}
        
        # Define a conditional edge to decide whether to continue or end the workflow
        def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
            messages = state["messages"]
            last_message = messages[-1]
            # If there is a tool call, then we finish
            if getattr(last_message, "tool_calls", None):
                return END
            if last_message.content.startswith("Error:"):
                return "query_gen"
            else:
                return "correct_query"

        # Done with all the setup, now add nodes and edges

        # Add nodes using the instance method
        workflow.add_node("first_tool_call", first_tool_call)
        workflow.add_node("list_tables_tool", create_tool_node([self.list_tables_tool]))
        workflow.add_node("get_schema_tool", create_tool_node([self.get_schema_tool]))
        workflow.add_node(
            "model_get_schema",
            lambda state: {
                "messages": [model_get_schema.invoke(
                    state["messages"] + [
                        ("user", "Include all possibly relevant tables. Particularly, consider the relationships among all the tables base on its meaning and foreign keys. What are the tables relevant to the question?")
                    ]
                )],
            },
        )
        workflow.add_node("query_gen", self.query_gen_node)
        workflow.add_node("correct_query", model_check_query)
        workflow.add_node("execute_query", create_tool_node([db_query_tool]))

        # Add edges
        workflow.add_edge(START, "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        # Add conditional edges
        workflow.add_conditional_edges("query_gen", should_continue)
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "query_gen")       
        
        # Store the workflow for later visualization
        self._workflow = workflow
        return workflow.compile()

    def draw_graph(self, draw_method="API", output_file="workflow_graph.png"):
        """Draw the agent's workflow graph using Mermaid and save to a file.
        
        Args:
            draw_method: The method to use for drawing. Can be "API" or other methods supported by Mermaid.
            output_file: Path to save the PNG file. Defaults to "workflow_graph.png".
            
        Returns:
            str: Path to the saved PNG file
        """
        # Get the uncompiled graph and draw it
        draw_method = getattr(MermaidDrawMethod, draw_method.upper(), MermaidDrawMethod.API)
        graph_data = self.app.get_graph().draw_mermaid_png(
            draw_method=draw_method,
            return_bytes=True  # Get the PNG data as bytes
        )
        
        # Save the PNG data to a file
        with open(output_file, 'wb') as f:
            f.write(graph_data)
            
        return output_file

    def invoke(self, question: str) -> Optional[str]:
        """Run the agent with a question and return the answer.
        
        Args:
            question: The natural language question to answer
            
        Returns:
            str: The formatted answer or None if processing fails
        """
        messages = self.app.invoke({"messages": [("user", question)]})
        json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
        return json_str
    
    def stream_invoke(self, question: str) -> Optional[str]:
        for event in self.app.stream(
            {"messages": [("user", question)]}
        ):
            print(event)
            print("\n")

class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")
