from llama_index.core.llms import ChatMessage
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import (
    load_index_from_storage,
    StorageContext
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import List
from llama_index.core.schema import Document
import os
# local
from htmlTemplates import *
from ai_support import *

class VectorIndexRetrieverWrapper(VectorIndexRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retrieve(query)

def initialize_conversation_chain(index, DEBUG=False):
    """
    Initialize the conversation chain using the loaded index and LLM.
    Adjust this function based on your specific conversation chain setup.
    """
    # Configure retriever with custom settings
    retriever = VectorIndexRetrieverWrapper(
        index=index,
        similarity_top_k=20,  # Customize as needed
    )
    
    if DEBUG:
        # Debug: Test retriever with a sample query
        sample_query = "test get_relevant_documents"
        sample_docs = retriever.get_relevant_documents(sample_query)
        st.write(f"Number of documents retrieved for sample query '{sample_query}': {len(sample_docs)}")
        
    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer()
    
    # Add a postprocessor
    similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)  # Lowered cutoff
    
    # Assemble the custom query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[similarity_postprocessor],
    )    
    return query_engine

def create_new_chat(directory, emb_option, llm_option, storage_context,
                    STORAGE_DIR, DEBUG=False):
    # Load documents from the specified directory
    print(f'Loading documents from {directory}')
    documents = SimpleDirectoryReader(directory).load_data()
    if not documents:
        st.warning("No documents found in the specified directory.")
        raise ValueError("No documents loaded.")

    # Set up embedding model
    embedding_model = get_embedding(emb_option=emb_option)

    Settings.embed_model = embedding_model
    if Settings.embed_model is None:
        st.error("Failed to initialize embedding model.")
        raise ValueError("Embedding model not initialized.")

    # Set up LLM
    Settings.llm = get_LLM(llm_option=llm_option, env_vars=os.environ)
    if Settings.llm is None:
        st.error("Failed to initialize LLM.")
        raise ValueError("LLM not initialized.")
    
    print('Loaded Embedding model and LLM')

    # Create the vector index with Chroma as vector store
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    st.session_state.index = index
    if DEBUG:
        # Debug: Check number of indexed documents
        st.write(f"Number of documents indexed: {len(documents)}")

    # Initialize the conversation chain with custom query engine
    print('Initializing conversation chain...')
    st.session_state.conversation_chain = initialize_conversation_chain(st.session_state.index, DEBUG=DEBUG)
    if st.session_state.conversation_chain is None:
        raise ValueError("Conversation chain initialization failed.")
    
    # Initialize chat_engine with the chat_memory
    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
        memory=st.session_state.chat_memory
    )
    
    # Persist the index to disk
    st.session_state.index.storage_context.persist(persist_dir=STORAGE_DIR)
    st.session_state.documents_loaded = True
    st.success("Documents loaded, index created, and persisted successfully!")

def load_prior_index(emb_option, llm_option, STORAGE_DIR, DEBUG=False):
    print('Loading StorageContext from defaults...')
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=STORAGE_DIR),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=STORAGE_DIR),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=STORAGE_DIR),
    )
    print('Loading embedding model and LLM...')
    embedding_model = get_embedding(emb_option=emb_option)
    Settings.embed_model = embedding_model
    Settings.llm = get_LLM(llm_option=llm_option, env_vars=os.environ)

    # Attempt to load an existing index
    print('Loading index from storage...')
    index = load_index_from_storage(storage_context)    

    st.session_state.index = index
    st.session_state.documents_loaded = True
    st.success("Persisted index loaded successfully!")        

    # Initialize the conversation chain with the loaded index
    print('Initializing conversation chain...')
    st.session_state.conversation_chain = initialize_conversation_chain(st.session_state.index, DEBUG=DEBUG)
    if st.session_state.conversation_chain is None:
        raise ValueError("Conversation chain initialization failed!")

    # Initialize chat_engine with the chat_memory
    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
        memory=st.session_state.chat_memory
    )

def load_user_documents(directory, emb_option, llm_option, 
                        STORAGE_DIR, DEBUG=False):
    # Try to load a persisted index first
    try:
        print('Running load_prior_index()')
        load_prior_index(emb_option, llm_option, STORAGE_DIR, DEBUG=DEBUG)

    except Exception as e:
        print('*** ***')
        st.warning("No existing index found. Creating a new one...")
        # If no persisted index, create a new one
        storage_context = StorageContext.from_defaults()  # create a fresh StorageContext
        print('Running create_new_chat')
        create_new_chat(directory, emb_option, llm_option, storage_context, STORAGE_DIR, DEBUG=DEBUG)


def conversation_html(user_input, chat_store_path, DEBUG=False):
    if st.session_state.chat_engine is None:
        raise ValueError("Chat engine is not initialized.")

    # Use the chat engine (with memory integration) to generate a response
    response = st.session_state.chat_engine.query(user_input)
    if DEBUG:
        st.write("Response Object:", response)
    # print('Question:')
    # print(user_input)
    # print('Response:')
    # print(response)
    answer = str(response)
    # print('Answer:')
    # print(answer)
    # Create ChatMessage instances
    user_message = ChatMessage(role="user", content=user_input)
    bot_message = ChatMessage(role="assistant", content=answer)  # Changed to "assistant"

    # Add messages to chat_memory
    st.session_state.chat_memory.put(user_message)
    st.session_state.chat_memory.put(bot_message)
    print("Messages stored in chat_memory.")

    # Persist the chat_store immediately after storing messages
    if st.session_state.chat_memory.chat_store.store:
        st.session_state.chat_memory.chat_store.persist(persist_path=chat_store_path)
        print("Chat store persisted successfully.")
    else:
        print("Chat store is empty; nothing to persist.")   
    
    if DEBUG:
        # Debug: Print chat_store after storing messages
        print("Chat Store after storing messages:", st.session_state.chat_memory.chat_store.store)
        st.write("Chat Store after storing messages:", st.session_state.chat_memory.chat_store.store)

def update_html():
    st.markdown("---")
    st.subheader("Chat History")
    
    # Access messages from chat_store using the chat_store_key
    messages = st.session_state.chat_memory.get_all()
    
    # Create a scrollable chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display messages in reverse order (newest first)
    for msg in reversed(messages):
        role = msg.role
        # import pdb
        # pdb.set_trace() 

        # Extract content from the blocks field
        content = ""
        if hasattr(msg, 'blocks') and isinstance(msg.blocks, list):
            content = " ".join([block.text for block in msg.blocks if getattr(block, 'block_type', '') == 'text'])
        else:
            content = msg.content

        if role == "user":
            st.markdown(user_template.format(user_icon=user_icon_base64, message=content), unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(bot_template.format(bot_icon=bot_icon_base64, message=content), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


