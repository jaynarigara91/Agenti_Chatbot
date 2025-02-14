from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class Chatbot:
    def __init__(self):
        self.llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b")

    def tool_call(self):
        tool = TavilySearchResults(max_results=2)
        tools = [tool]
        self.tool_node = ToolNode(tools=tools)
        self.llm_with_tool = self.llm.bind_tools(tools)

    def call_model(self, state: MessagesState):
        messages = state['messages']
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [response]}

    def route_function(self, state: MessagesState):
        message = state['messages'][-1]
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return "tools"
        return END

    def __call__(self):
        self.tool_call()
        workflow = StateGraph(MessagesState)
        workflow.add_node("Agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "Agent")
        workflow.add_conditional_edges(
            'Agent',
            self.route_function,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "Agent")
        self.app = workflow.compile(checkpointer=MemorySaver())
        return self.app

bot = Chatbot()
app = bot()

st.title("Agentic Chatbot 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
   
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        config = {"configurable": {"thread_id": "2"}}
        response = app.invoke({'messages': [{"role": "user", "content": prompt}]}, config=config)
        full_response = response['messages'][-1].content

        
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
