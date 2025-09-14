from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()
from pinecone import ServerlessSpec
from pinecone import Pinecone
from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_docs = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_docs)
texts_chunk = text_split(minimal_docs)
embeddings = download_embeddings()

pinecode_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecode_api_key)

index_name = "med-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )

index = pc.Index(index_name)

# Reuse existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Estimate KB size of a document (text + metadata)
def get_doc_size_kb(doc):
    return (len(str(doc.page_content)) + len(str(doc.metadata))) / 1024

# Batching parameters
MAX_DOCS_PER_BATCH = 40
MAX_BATCH_SIZE_KB = 1800  # Stay well below 2MB limit

current_batch = []
current_batch_size_kb = 0

for doc in tqdm(texts_chunk, desc="Uploading to Pinecone"):
    doc_size_kb = get_doc_size_kb(doc)

    # Optional: log unusually large chunks
    if doc_size_kb > 100:
        print(f"⚠️ Warning: Large doc chunk ({doc_size_kb:.2f} KB)")

    # If adding this doc would exceed limits, upload current batch
    if (len(current_batch) >= MAX_DOCS_PER_BATCH) or (current_batch_size_kb + doc_size_kb > MAX_BATCH_SIZE_KB):
        try:
            docsearch.add_documents(current_batch)
        except Exception as e:
            print(f"❌ Error uploading batch: {e}")
        current_batch = []
        current_batch_size_kb = 0

    # Add doc to current batch
    current_batch.append(doc)
    current_batch_size_kb += doc_size_kb

# Final batch upload
if current_batch:
    try:
        docsearch.add_documents(current_batch)
    except Exception as e:
        print(f"❌ Error uploading final batch: {e}")
        
print("All documents uploaded successfully to Pinecone!")
