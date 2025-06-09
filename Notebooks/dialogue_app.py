import os
import faiss
import gradio as gr
import pickle
import random
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# === STEP 0: Your raw dialogue data ===
# Replace this with your actual dialogue_history variable
dialogue_history = [
    # Example structure:
    # {'speaker': '🔴 Boot & Shoe Union (1920s)', 'query': 'example prompt', 'response': 'example reply', 'context': '...'},
]

# === STEP 1: Convert to DataFrame with speaker-labeled conversation ===
rows = []
for i in range(0, len(dialogue_history), 5):
    chunk = dialogue_history[i:i+5]
    if len(chunk) == 5:
        initial_prompt = chunk[0]['query']
        full_convo = '\n'.join([f"{entry['speaker']}: {entry['response']}" for entry in chunk])
        rows.append({'initial_prompt': initial_prompt, 'full_conversation': full_convo})

df = pd.DataFrame(rows)

# === STEP 2: Create FAISS index from initial_prompt ===
model = SentenceTransformer('all-mpnet-base-v2')
initial_prompts = df['initial_prompt'].tolist()
embeddings = model.encode(initial_prompts, show_progress_bar=True, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === STEP 3: Save FAISS index + metadata ===
faiss_index_path = 'initial_prompt.index'
metadata_path = 'prompt_index_map.pkl'

faiss.write_index(index, faiss_index_path)
with open(metadata_path, 'wb') as f:
    pickle.dump(df.to_dict(orient='records'), f)

# === STEP 4: Gradio app ===
# Reload for clarity (optional if running as one script)
index = faiss.read_index(faiss_index_path)
with open(metadata_path, 'rb') as f:
    prompt_data = pickle.load(f)

def format_conversation_only(row):
    lines = row['full_conversation'].split('\n')
    return "\n\n".join([f"> {line}" for line in lines])

def search_conversations(query, top_k=1):
    if not query.strip():
        return "❌ Please enter a query."
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k=top_k)
    top = prompt_data[I[0][0]]
    return format_conversation_only(top)

def random_conversation():
    rand = random.choice(prompt_data)
    return format_conversation_only(rand)

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ LLM Dialogue Explorer")
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Search", placeholder="e.g. strikes, housing, gig economy...")
            search_btn = gr.Button("🔍 Search")
            random_btn = gr.Button("🎲 Random Conversation")
        with gr.Column():
            output_box = gr.Markdown()

    search_btn.click(fn=search_conversations, inputs=query_input, outputs=output_box)
    random_btn.click(fn=random_conversation, outputs=output_box)

demo.launch()
