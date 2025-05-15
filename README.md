# PDF Question Answering App

This is a FastAPI-based web service that lets you upload a PDF document and ask questions about its content. It uses AI models to understand and answer your queries based on the PDF text.

## Features

- Upload PDF files
- Automatically extract and chunk text from PDFs
- Embed chunks using sentence transformers for semantic search
- Ask questions related to the uploaded PDF
- Get answers from a question-answering AI model (RoBERTa)

## Technologies Used

- FastAPI — backend web framework  
- PyMuPDF (fitz) — to extract text from PDF files  
- Sentence Transformers — for embedding text chunks  
- FAISS — for fast similarity search  
- Hugging Face Transformers — question-answering pipeline  
- CORS Middleware — to allow frontend access

## How to Run Locally

1. Clone this repo:
