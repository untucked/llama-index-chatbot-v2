# ai_support.py

from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.embeddings import OpenAIEmbeddings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
    
def get_embedding(emb_option='stella_lite'):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    if emb_option == 'bge-base-en':
        model_name = "BAAI/bge-base-en-v1.5"        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    elif emb_option == 'stella':
        model_name = "blevlabs/stella_en_v5"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    elif emb_option == 'stella_lite':
        model_name = 'dunzhang/stella_en_400M_v5'
        model_kwargs["trust_remote_code"] = True  # Add this to the model_kwargs
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif emb_option == 'MiniLM-L6':
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        model_kwargs["trust_remote_code"] = True  # Add this to the model_kwargs
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif emb_option == 'Ollama':
        embeddings = OllamaEmbedding(
            model_name="llama2",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
    elif emb_option == 'OpenAI':
        embeddings = OpenAIEmbeddings()
    else:
        raise ValueError("Embeddings option must be provided.")

    if emb_option not in ('Ollama', 'OpenAI'):
        print('Applying LangchainEmbedding()')
        embeddings = LangchainEmbedding(embeddings)

    return embeddings

def get_LLM(llm_option='Ollama3.2', env_vars=None):
    huggingfacehub_api_token = env_vars.get("HUGGINGFACEHUB_API_TOKEN", "")
    MODEL_CONFIG = {
        'OpenAI': {'type': 'OpenAI', 'path': None},
        'meta-llama': {'type': 'HuggingFace', 'path': "meta-llama/Llama-2-7b-chat-hf"},
        'Qwen': {'type': 'HuggingFace', 'path': "Qwen/Qwen2.5-1.5B"},
        'MiniChat': {'type': 'HuggingFace', 'path': "GeneZC/MiniChat-1.5-3B"},
        'sentence-transformers': {'type': 'HuggingFace', 'path': "sentence-transformers/all-MiniLM-L6-v2"},
        'Ollama3.2': {'type': 'Ollama', 'path': 'llama3.2'},
        'Ollama3': {'type': 'Ollama', 'path': 'llama3'}
    }
    model_config = MODEL_CONFIG.get(llm_option)
    if model_config is None:
        raise ValueError(f"Unsupported chat_ai: {llm_option}")
    # Initialize the appropriate LLM based on model type
    if model_config['type'] == 'OpenAI':
        from llama_index.llms.openai import ChatOpenAI  # Ensure this import is correct
        llm = ChatOpenAI(api_key=env_vars.get("OPENAI_API_KEY", ""))
    elif model_config['type'] == 'HuggingFace':
        from langchain_huggingface import HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(
            repo_id=model_config['path'],
            temperature=0.5,  # Specify temperature directly
            max_length=128,   # You can include other parameters here
            huggingfacehub_api_token=huggingfacehub_api_token,
            timeout=1000       # Increase the timeout as needed
        )
    elif model_config['type'] == 'Ollama':
        llm = Ollama(model=model_config['path'], request_timeout=360.0)
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    return llm

