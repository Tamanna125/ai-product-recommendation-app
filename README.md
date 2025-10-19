# 🛋️ AI Product Recommender & Analytics App

A full-stack web application demonstrating the integration of Machine Learning (NLP, CV) and Generative AI for product recommendations and analytics, built within a 2-day timeframe. The application recommends furniture products based on user prompts, generates creative AI descriptions, and displays analytics about the product dataset.

## ✨ Features

* **AI-Powered Recommendations:** Uses text embeddings (`sentence-transformers`) and a vector database (Pinecone) for semantic search based on user prompts.
* **Generative Descriptions:** Employs a lightweight Generative AI model (`google/flan-t5-small`) via LangChain to create unique, engaging descriptions for recommended products.
* **Computer Vision Demonstration:** Includes a notebook (`Model_Training.ipynb`) showing how to fine-tune a pre-trained ResNet-18 model for image classification on product categories.
* **Interactive Frontend:** A simple React interface for users to input prompts and view recommendations.
* **Analytics Dashboard:** A separate page displaying key statistics and visualizations (e.g., top categories, top brands) derived from the product dataset.
* **Modern Backend:** Built with FastAPI for high performance.

---

## 💻 Tech Stack

* **Backend:** Python, FastAPI
* **Frontend:** React (Vite), JavaScript, CSS, Axios, Recharts
* **ML/NLP:** `sentence-transformers`, `transformers` (Hugging Face)
* **CV:** `torchvision` (PyTorch)
* **Vector Database:** Pinecone
* **GenAI Framework:** LangChain
* **Environment:** Python Virtual Environment, Node.js

---

## 📁 Project Structure

product-recommendation-app/
├── backend-env/          # Python virtual environment
├── data/                 # Sample images downloaded for CV training
│   └── cv_training/
├── frontend/             # React frontend application (Vite)
│   ├── public/
│   └── src/              # React components and CSS
├── Data_Analytics.ipynb  # Jupyter Notebook for Exploratory Data Analysis
├── Model_Training.ipynb  # Jupyter Notebook for ML/NLP/CV model pipelines
├── cv_classifier_model.pth # Saved weights for the trained CV model
├── main.py               # FastAPI backend server code
├── products_dataset.csv  # The provided dataset
├── requirements.txt      # Python dependencies for the backend
├── README.md             # This file
└── ...                   # Other config files (package.json, etc.)

# 🛋️ AI Product Recommender & Analytics App

A full-stack web application demonstrating the integration of Machine Learning (NLP, CV) and Generative AI for product recommendations and analytics, built within a 2-day timeframe. The application recommends furniture products based on user prompts, generates creative AI descriptions, and displays analytics about the product dataset.

## ✨ Features

* **AI-Powered Recommendations:** Uses text embeddings (`sentence-transformers`) and a vector database (Pinecone) for semantic search based on user prompts.
* **Generative Descriptions:** Employs a lightweight Generative AI model (`google/flan-t5-small`) via LangChain to create unique, engaging descriptions for recommended products.
* **Computer Vision Demonstration:** Includes a notebook (`Model_Training.ipynb`) showing how to fine-tune a pre-trained ResNet-18 model for image classification on product categories.
* **Interactive Frontend:** A simple React interface for users to input prompts and view recommendations.
* **Analytics Dashboard:** A separate page displaying key statistics and visualizations (e.g., top categories, top brands) derived from the product dataset.
* **Modern Backend:** Built with FastAPI for high performance.

---

## 💻 Tech Stack

* **Backend:** Python, FastAPI
* **Frontend:** React (Vite), JavaScript, CSS, Axios, Recharts
* **ML/NLP:** `sentence-transformers`, `transformers` (Hugging Face)
* **CV:** `torchvision` (PyTorch)
* **Vector Database:** Pinecone
* **GenAI Framework:** LangChain
* **Environment:** Python Virtual Environment, Node.js

---

## 📁 Project Structure