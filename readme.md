# Llama Index Chatbot

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-blue.svg)

> You can find the tutorial for this project on [YouTube] in the future if there's enough interest (TBD)

## Introduction
------------
The **Llama Index Chatbot App (v2)** is a Python-based application that enables interactive conversations with multiple PDF documents. Users can pose questions in natural language, and the chatbot provides relevant answers derived from the content of the loaded PDFs. This tool leverages advanced language models to ensure accurate and contextually appropriate responses. Note that the chatbot's knowledge is confined to the uploaded documents.
Differences from v1:
- **Saved chat history** Save chat history using llama_index SimpleChatStore 
- **Load preloaded Indexes** Save the prior loaded index values 
Goals for v3:
- **Make sure the embedded model is the same as the prior index** having a differenct embedded model would throw an error
- **Clear Index and reload documents** Save the prior loaded index values 

### Goal
------------
I wanted to create a pdf reader to run locally on a laptop and utilize local LLM's such as Llama and Ollama. 
There's an option for OpenAI as well so they handle all the heavy lifting, but I wanted to try the other options as well.
The ideal user for this would be 
- someone who wants to read PDF's with LLM's locally & 
- someone who has limited admin access to their laptop and are required to read lots of PDF's

## How It Works
------------

The application operates through the following steps:

1. **Directory Loading:** The app scans a specified directory to read multiple PDF documents and extracts their textual content.
2. **Text Chunking:** The extracted text is segmented into manageable chunks to facilitate efficient processing.
3. **Language Model Integration:** Utilizes a language model to generate vector representations (embeddings) of the text chunks.
4. **Similarity Matching:** When a user asks a question, the app compares it against the text chunks to identify the most semantically similar sections.
5. **Response Generation:** The selected text chunks are fed into the language model to generate a coherent and relevant response based on the PDF content.

## Features
------------
- **Multiple PDF Support:** Load and interact with numerous PDF documents simultaneously.
- **Natural Language Understanding:** Ask questions in plain English and receive precise answers.
- **Chat History:** Maintain a record of all interactions for easy reference.
- **Efficient Indexing:** Rapid setup and indexing for quick query responses.
- **Scalable Embeddings:** Utilizes robust embedding models to understand and process text effectively.

## Dependencies and Installation
----------------------------
Follow these steps to set up the Llama Index Chatbot App on your local machine:

### **1. Clone the Repository**
git clone https://github.com/untucked/llama-index-chatbot-v2.git
cd llama-index-chatbot-v2

### **2. Run the Application
bash
streamlit run main.py

### **3. Interact with the App

The application will open in your default web browser.
Select the embedding option and LLM option. The Embedding reads the PDFs and the LLM interprets the PDFs with your questions and provides an answer.
Load Documents: Enter the directory path containing your PDF documents and click "Load Documents."
Chat: Once documents are loaded, use the chat interface to ask questions related to the content of the PDFs.
