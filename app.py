import fitz  # PyMuPDF
import openai
from openai import OpenAI
import re
import json
import os
import numpy as np
from sklearn.cluster import KMeans
import time
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS

load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
CORS(app) # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(basedir, 'outputs')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_length=500):
    text = re.sub(r'\n+', '\n', text.strip())
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
    
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) <= max_length:
            current_chunk += " " + p
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(text_chunks, batch_size=10, max_retries=3, wait_time=2):
    embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                embeddings.extend([item.embedding for item in response.data])
                break  # success, exit retry loop
            except Exception as e:
                print(f"Error on batch {i//batch_size + 1} attempt {attempt+1}: {e}")
                time.sleep(wait_time * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to get embeddings for batch starting at index {i}")
    return np.array(embeddings)

def cluster_chunks(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels

def group_by_cluster(chunks, labels):
    clustered = {}
    for i, label in enumerate(labels):
        clustered.setdefault(int(label), []).append(chunks[i])
    return clustered

def generate_prompt(chunk):
    prompt = f"""Given the following text:

\"\"\"{chunk}\"\"\"

Write a clear, natural-language question a user might ask that this text answers."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant creating training data for fine-tuning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def create_finetune_dataset(clustered_chunks, output_path="fine_tune_data.jsonl"):
    with open(output_path, 'w', encoding='utf-8') as f:
        for cluster in clustered_chunks.values():
            combined = " ".join(cluster)
            question = generate_prompt(combined)
            entry = {
                "prompt": question,
                "completion": combined.strip() + "###"
            }
            json.dump(entry, f)
            f.write('\n')
    return output_path

def process_pdf_to_jsonl(pdf_path, output_path="fine_tune_data.jsonl", n_clusters=5):
    print("🔍 Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    print("✂️ Chunking text...")
    chunks = chunk_text(text)
    print(f"📝 Number of chunks created: {len(chunks)}")
    print("📐 Getting embeddings in batches...")
    embeddings = get_embeddings(chunks)
    print("🔄 Clustering chunks...")
    labels = cluster_chunks(embeddings, n_clusters=n_clusters)
    clustered_chunks = group_by_cluster(chunks, labels)
    print("🧠 Generating prompts and writing JSONL file...")
    return create_finetune_dataset(clustered_chunks, output_path)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        filename = file.filename
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)

        n_clusters = int(request.form.get('n_clusters', 7))
        output_filename = os.path.splitext(filename)[0] + '.jsonl'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        try:
            process_pdf_to_jsonl(pdf_path, output_path, n_clusters=n_clusters)
            return jsonify({
                "download_url": f"/download/{output_filename}",
                "filename": output_filename
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 