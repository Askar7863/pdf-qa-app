from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

pdf_chunks = []
pdf_embeddings = None

# Create overlapping chunks
def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_chunks, pdf_embeddings

    # Read PDF content
    doc = fitz.open(stream=await file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Chunk and embed
    pdf_chunks = create_chunks(full_text)
    embeddings = embedder.encode(pdf_chunks)
    pdf_embeddings = faiss.IndexFlatL2(embeddings.shape[1])
    pdf_embeddings.add(np.array(embeddings))

    return {"message": "PDF uploaded and indexed successfully!"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global pdf_chunks, pdf_embeddings

    if not pdf_chunks or pdf_embeddings is None:
        return JSONResponse(content={"error": "No PDF uploaded yet."}, status_code=400)

    # Embed and search
    question_embedding = embedder.encode([question])
    D, I = pdf_embeddings.search(np.array(question_embedding), k=3)

    # Combine top chunks for better QA
    context = " ".join([pdf_chunks[i] for i in I[0]])
    answer = qa_pipeline(question=question, context=context)

    return {"answer": answer["answer"]}

