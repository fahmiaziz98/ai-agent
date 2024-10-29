from src.agent import SQLAgentRAG
from src.tools import retriever
from langchain_groq import ChatGroq
from src.constant import GROQ_API_KEY, CONFIG

Q1 = "How many different aircraft models are there in the aircrafts_data table? and what are the models?"

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.1,
    verbose=True
)

agent = SQLAgentRAG(llm=llm, tools=retriever)
# query = "Hi, my name Fahmi, nice to meet you"
for events in agent.graph.stream(
    {"messages": [("user", Q1)]}, CONFIG, stream_mode="values"
):
    events["messages"][-1].pretty_print()