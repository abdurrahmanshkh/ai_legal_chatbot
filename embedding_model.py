from langchain_huggingface import HuggingFaceEmbeddings

# Now you can initialize your embedding model:
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
