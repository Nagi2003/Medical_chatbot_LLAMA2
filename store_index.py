from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PC
from pinecone import Pinecone,ServerlessSpec
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#print(PINECONE_API_KEY)
#print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name='medical'

pc = Pinecone(api_key="19117b97-7e0a-4546-ba88-0f5ebdd0615d")

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
  name="medical",
  dimension=384,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)

index = pc.Index(index_name)
index.describe_index_stats()

# Prepare document chunks
docs_chunks = [t.page_content for t in text_chunks]

# Create Pinecone index using langchain (assuming Langchain uses the existing index)
vectorstore= PC.from_texts(
    docs_chunks,
    embeddings,
    index_name="medical"
)