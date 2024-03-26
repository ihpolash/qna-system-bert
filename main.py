import streamlit as st
from PyPDF2 import PdfReader
import re
import numpy as np
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from annoy import AnnoyIndex
import torch

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Initialize Annoy index
vector_dim = 768  # Dimensionality of BERT embeddings
annoy_index = AnnoyIndex(vector_dim, 'angular')

# Initialize global variable for text chunks
text_chunks = []

def preprocess_text(text):
    # Remove special characters and lowercase the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    
    return text

def split_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) < chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_vectors(text_chunks, batch_size=32):
    vectors = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        
        # Tokenize the batch of text chunks
        encoded_inputs = bert_tokenizer.batch_encode_plus(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate vector representations using BERT
        with torch.no_grad():
            outputs = bert_model(**encoded_inputs)
            batch_vectors = outputs.last_hidden_state[:, 0, :].numpy()
        
        vectors.extend(batch_vectors)
    
    return vectors

def answer_question(question, top_k=5):
    # Generate vector representation for the question
    question_vector = generate_vectors([question])[0]
    
    # Perform similarity search using Annoy
    similar_indices = annoy_index.get_nns_by_vector(question_vector, top_k)
    
    # Retrieve the most relevant text chunks
    relevant_chunks = [text_chunks[idx] for idx in similar_indices]
    
    # Tokenize the question and relevant chunks
    encoded_inputs = bert_tokenizer.batch_encode_plus(
        [(question, chunk) for chunk in relevant_chunks],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Apply question-answering model to the relevant chunks
    with torch.no_grad():
        outputs = qa_model(**encoded_inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
    
    # Find the chunk with the highest answer score
    chunk_scores = start_scores.max(dim=1).values + end_scores.max(dim=1).values
    best_chunk_idx = chunk_scores.argmax().item()
    
    # Extract the answer from the best chunk
    start_idx = start_scores[best_chunk_idx].argmax().item()
    end_idx = end_scores[best_chunk_idx].argmax().item()
    answer = bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(encoded_inputs["input_ids"][best_chunk_idx][start_idx:end_idx+1]))
    
    return answer

def main():
    global text_chunks
    
    st.title("PDF Question Answering System")

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Read PDF file
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Preprocess and split text into chunks
        preprocessed_text = preprocess_text(text)
        text_chunks = split_text(preprocessed_text)

        # Generate vector representations
        vectors = generate_vectors(text_chunks)

        # Build Annoy index
        for i, vector in enumerate(vectors):
            annoy_index.add_item(i, vector)
        annoy_index.build(10)  # Number of trees

        st.success("PDF document processed and indexed successfully!")

    # Get user question
    question = st.text_input("Enter your question")

    if question:
        # Answer the question
        answer = answer_question(question)
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()