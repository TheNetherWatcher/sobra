import os
import sys
sys.path.insert(0, '..')
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "soybean.txt")

print(file_path)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  
)

Settings.llm = HuggingFaceLLM(
    model_name="HuggingFaceTB/SmolLM-360M-Instruct",
    tokenizer_name="HuggingFaceTB/SmolLM-360M-Instruct",
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
    device_map="auto"
)

documents = SimpleDirectoryReader( input_files=[file_path]).load_data()
document = Document(text="\n\n".join([doc.text for doc in documents]))

def rag(prompt):     
    node_parser = SimpleNodeParser.from_defaults(chunk_size=64,chunk_overlap=2)
    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)

    return response