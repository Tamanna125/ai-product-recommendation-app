# --- [Part 1: Imports] ---
# Backend server
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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Type hints for request bodies
from pydantic import BaseModel

# --- [Part 2: Initialization] ---
# This code runs ONCE when the server starts.

# 1. Initialize FastAPI app
app = FastAPI(title="Product Recommendation API")

# 2. Add CORS middleware
# This allows your React app (running on http://localhost:5173) to make requests to this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # The origin of your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the dataset (for analytics)
try:
    df = pd.read_csv('products_dataset.csv')
    df = df.fillna("") # Clean missing values
    
    # --- Re-using our cleaning function from the notebook ---
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
            
    df['clean_category'] = df['categories'].apply(clean_category)
    # --- End of cleaning function ---

except FileNotFoundError:
    print("WARNING: 'products_dataset.csv' not found. Analytics endpoint will be empty.")
    df = pd.DataFrame()

# 4. Load the NLP Embedding Model
print("Loading Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding Model loaded.")

# 5. Connect to Pinecone
# --- !!! IMPORTANT !!! ---
# --- REPLACE WITH YOUR VALUES ---
PINECONE_API_KEY = "pcsk_7Ey3x5_ULhsd8bPjwRmcB1tuygNmR2dv35axU9hBxsWLBdUReg4qm4NrHKkeQ5QJdDsGDV"
INDEX_NAME = "product-recommender"

if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_API_KEY_HERE":
    raise EnvironmentError("PINECONE_API_KEY not set in main.py. Please add it.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index '{INDEX_NAME}'.")


# 6. Load GenAI Model & LangChain (Requirement 4 & 5) [cite: 31, 46]
print("Loading GenAI Model (This may take a few minutes)...")

# We use a lightweight, fast model for creative descriptions [cite: 31]
model_id = "distilgpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
genai_model = AutoModelForCausalLM.from_pretrained(model_id)

# Set up the pipeline
pipe = pipeline(
    "text-generation",
    model=genai_model,
    tokenizer=tokenizer,
    max_new_tokens=50 # Keep descriptions short
)

# Connect the model to LangChain [cite: 46]
llm = HuggingFacePipeline(pipeline=pipe)

# Define our prompt template
template = """
You are a creative marketing assistant. Write a short, engaging product description for the following item:
Product: {product_details}
Description:
"""
prompt_template = PromptTemplate(input_variables=["product_details"], template=template)
description_chain = LLMChain(llm=llm, prompt=prompt_template)

print("GenAI Model and LangChain pipeline loaded.")


# --- [Part 3: API Endpoints] ---

# Define the Pydantic model for our request body
class RecommendRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    """A simple root endpoint to check if the server is running."""
    return {"message": "API server is running."}


@app.post("/recommend")
def post_recommendations(request: RecommendRequest):
    """
    Main recommendation endpoint. [cite: 7]
    1. Takes a user's text prompt.
    2. Converts it to an embedding.
    3. Queries Pinecone for the top 5 most similar products.
    4. Generates a new description for each.
    5. Returns the results.
    """
    try:
        # 1. Convert user's prompt to an embedding
        query_embedding = embedding_model.encode(request.prompt).tolist()

        # 2. Query Pinecone [cite: 32]
        query_results = index.query(
            vector=query_embedding,
            top_k=5, # Get the top 5 results
            include_metadata=True
        )

        recommendations = []
        for match in query_results.get('matches', []):
            product = match.get('metadata', {})
            
            # 3. Generate a new description using GenAI [cite: 31]
            product_details = f"Title: {product.get('title')}, Brand: {product.get('brand')}, Categories: {product.get('categories')}"
            
           # Run the LangChain chain
            output_dict = description_chain.invoke({"product_details": product_details})
            
            # Clean up the generated text (the output is now in a 'text' key)
            generated_description = output_dict['text'].strip().split('\n')[0]

            # 4. Add to our results
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
    Analytics endpoint for the dashboard. [cite: 8, 36]
    Reads the CSV and calculates some simple stats.
    """
    if df.empty:
        return {"error": "Dataset not loaded."}
        
    try:
        # 1. Price distribution
        price_dist = df['price'].dropna().describe().to_dict()
        
        # 2. Top 10 categories (using our clean column)
        top_categories = df['clean_category'].value_counts().nlargest(10).to_dict()

        # 3. Top 10 brands
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