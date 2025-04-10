from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict

# FAQ Data
faq_data = {
    "What is LangGraph?": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    "How do I install LangGraph?": "You can install LangGraph using pip: pip install langgraph",
    "Is LangGraph open source?": "Yes, LangGraph is open source and available on GitHub.",
    "What is an LLM?": "LLM stands for Large Language Model, a type of AI model designed to understand and generate human-like text.",
    "Can I use LangGraph with OpenAI?": "Yes, LangGraph can be integrated with OpenAI's language models for conversational workflows.",
}

formatted_faq = "\n".join([f"Q: {q}\nA: {a}" for q, a in faq_data.items()])

template = PromptTemplate.from_template(
    """You are a helpful FAQ bot. Here are some FAQs:

{faq}

User question: {question}

Return the most relevant answer. If not found, say you don't know.
"""
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
faq_chain: Runnable = template | llm | StrOutputParser()


# Define schema for LangGraph
class FAQState(TypedDict):
    question: str
    answer: str


# Node function
def retrieve_answer(state: FAQState) -> FAQState:
    question = state["question"]
    answer = faq_chain.invoke({"faq": formatted_faq, "question": question})
    return {"question": question, "answer": answer}


# Build graph
builder = StateGraph(state_schema=FAQState)
builder.add_node("answer_faq", retrieve_answer)
builder.set_entry_point("answer_faq")
builder.set_finish_point("answer_faq")
faq_graph = builder.compile()

# Chat loop
if __name__ == "__main__":
    print("Welcome to the LangGraph FAQ bot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break
        result = faq_graph.invoke({"question": user_input, "answer": ""})
        print("Bot:", result["answer"])
