import asyncio
import os
import ssl
from typing import List, Any, Dict

import certifi
from chromadb.utils import results
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


from logger import Colors, log_header, log_error, log_info, log_success, log_warning

load_dotenv()  # Load environment variables from .env file

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorStore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split urls into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    res = tavily_map.invoke("https://python.langchain.com/")

    log_success(
        f"TavilyCrawl: Successfully crawled {len(res)} URLs from documentation site"
    )

    # Split urls into batches of 20
    url_batches = chunk_urls(list(res["results"]), chunk_size=20)

    log_info(
        f"URL Processing Split {len(res['results'])} URLs into {len(url_batches)} batches",
        Colors.BLUE,
    )


if __name__ == "__main__":
    asyncio.run(main())
