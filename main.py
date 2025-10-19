# --- [Part 1: Imports] ---
# Backend server
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # To allow our React app to talk to this server

# Data handling
import pandas as pd
import ast

# Pinecone (Vector DB)
from pinecone import Pinecone

# NLP Model (Embeddings)
from sentence_transformers import SentenceTransformer

# GenAI & LangChain
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # For LCEL

# Type hints for request bodies
from pydantic import BaseModel


# --- [Part 2: Initialization] ---
app = FastAPI(title="Product Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # The origin of your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define cleaning function globally ---
def clean_category(cat_str):
    try:
        cat_list = ast.literal_eval(cat_str)
        if isinstance(cat_list, list) and len(cat_list) > 1:
            return cat_list[1]
        elif isinstance(cat_list, list) and len(cat_list) > 0:
            return cat_list[-1]
        else:
            return "Uncategorized"
    except (ValueError, SyntaxError):
        return cat_str if pd.notna(cat_str) else "Uncategorized"

# 3. Load dataset
try:
    df = pd.read_csv('products_dataset.csv')
    df = df.fillna("")
    # Apply cleaning function
    df['clean_category'] = df['categories'].apply(clean_category)

except FileNotFoundError:
    print("WARNING: 'products_dataset.csv' not found. Analytics endpoint will be empty.")
    df = pd.DataFrame()


# 4. Embedding model
print("Loading Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding Model loaded.")


# 5. Pinecone
# --- !!! IMPORTANT: REPLACE WITH YOUR *NEW* API KEY !!! ---
PINECONE_API_KEY = os.getenv("pcsk_7Ey3x5_ULhsd8bPjwRmcB1tuygNmR2dv35axU9hBxsWLBdUReg4qm4NrHKkeQ5QJdDsGDV")
INDEX_NAME = "product-recommender"

if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_NEW_API_KEY_HERE":
    raise EnvironmentError("PINECONE_API_KEY not set in main.py. Please add it.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index '{INDEX_NAME}'.")


# 6. GenAI + LangChain
print("Loading GenAI Model (This may take a few minutes)...")

model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
genai_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation",
    model=genai_model,
    tokenizer=tokenizer,
    max_new_tokens=50
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Corrected LangChain section using LCEL ---
template = """
Write a short, engaging product description for the following item:
Product: {product_details}
Description:
"""
prompt_template = PromptTemplate(input_variables=["product_details"], template=template)

# LCEL pipeline replaces LLMChain
# This chains the prompt, the model, and an output parser together
description_chain = prompt_template | llm | StrOutputParser()
# --- End of corrected LangChain section ---

print("GenAI Model and LangChain pipeline loaded.")


# --- [Part 3: API Endpoints] ---

class RecommendRequest(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    return {"message": "API server is running."}


@app.post("/recommend")
def post_recommendations(request: RecommendRequest):
    """
    Main recommendation endpoint.
    """
    try:
        # 1. Encode user query
        query_embedding = embedding_model.encode(request.prompt).tolist()

        # 2. Query Pinecone
        query_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        recommendations = []
        for match in query_results.get('matches', []):
            product = match.get('metadata', {})

            product_details = f"Title: {product.get('title')}, Brand: {product.get('brand')}, Categories: {product.get('categories')}"

            # --- Run the new LCEL chain (returns plain string) ---
            generated_description = description_chain.invoke({"product_details": product_details}).strip().split('\n')[0]
            # --- End of LCEL chain usage ---

            recommendations.append({
                "id": match.get('id'),
                "score": match.get('score'),
                "title": product.get('title'),
                "price": product.get('price'),
                "image_url": product.get('image_url'),
                "generated_description": generated_description
            })

        return {"recommendations": recommendations}

    except Exception as e:
        print(f"Error in /recommend: {e}")
        return {"error": str(e)}, 500


@app.get("/analytics")
def get_analytics():
    """
    Analytics endpoint for the dashboard.
    Reads the CSV and calculates some simple stats.
    """
    if df.empty:
        return {"error": "Dataset not loaded."}
        
    try:
        # We need the clean category applied here if not done globally
        if 'clean_category' not in df.columns:
             df['clean_category'] = df['categories'].apply(clean_category)

        # --- Corrected Price Cleaning ---
        # 1. Clean the 'price' column: remove '$' and ',' and convert to numeric
        clean_prices = pd.to_numeric(
            df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False),
            errors='coerce'
        )
        
        # 2. Now, describe the *cleaned* prices
        price_stats = clean_prices.dropna().describe()
        # --- End Price Cleaning ---

        # 3. Price distribution
        price_dist = {
            "count": int(price_stats.get("count", 0)),
            "mean": float(price_stats.get("mean", 0.0)),
        }
        
        # 4. Top 10 categories (using our clean column)
        top_categories = df['clean_category'].value_counts().nlargest(10).to_dict()

        # 5. Top 10 brands
        top_brands = df['brand'].value_counts().nlargest(10).to_dict()

        return {
            "total_products": int(len(df)),
            "price_distribution": price_dist,
            "top_categories": top_categories,
            "top_brands": top_brands
        }
    except Exception as e:
        print(f"Error in /analytics: {e}")
        return {"error": str(e)}, 500