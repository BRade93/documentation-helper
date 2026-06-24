import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)

# Initialize chat model
model = init_chat_model("qwen3:4b", model_provider="ollama")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve documentation to help answer user queries about LangChain"""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str):
    """Run the RAG retrieval pipeline
    Args:
        query (str): The users question
    Returns:
        Dictionary containing:
            -answer
            -context
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions abour LangChain documentation."
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the asnwer in the retrieved documentation say so. "
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs,
    }


if __name__ == "__main__":
    result = run_llm(
        query="What are deep agents? Use langchains official documentation"
    )
    print(result)
