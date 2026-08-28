# FlyRank Content Intelligence Platform 🚀

An end-to-end Machine Learning, RAG, and Agentic AI platform built for SEO content optimization. 

This repository upgrades a traditional data science capstone project into a production-grade **MLOps + RAG + Full-Stack** application. It automates the process of identifying decaying content, predicting refresh priorities using Machine Learning, and generating specific SEO recommendations using a Retrieval-Augmented Generation (RAG) agent.

---

## 🌟 What Makes This Project Special?

Unlike typical Jupyter Notebook student projects, this platform is built like a **production system**:

| Typical ML Project | This Project |
| :--- | :--- |
| Train a model in a Notebook | Automated feature pipeline & data leakage audits |
| Static CSV predictions | Live webpage analysis through a FastAPI REST backend |
| No Explainability | Reason codes + RAG AI Playbook suggestions |
| No UI / Deployment | Premium React Dashboard + Docker orchestration |
| No Monitoring | MLflow experiment tracking & Evidently data drift reports |
| Simple Prompts | RAG Pipeline utilizing ChromaDB + Live URL scraping |

---

## 🏗️ System Architecture

### 1. Data Engineering & ML Pipeline
- **Feature Engineering**: Derives advanced SEO metrics (e.g., CTR Delta, Impression Growth, Freshness & Decay scores, Opportunity scores).
- **Leakage Audit**: Automatically splits data using a time-based approach (`content_age_days`) to prevent future-metric leakage.
- **ML Training**: Trains Random Forest, Gradient Boosting, and XGBoost models on refresh scores and action classifications. The best model is chosen based on MAE.
- **MLOps**: Uses **MLflow** for parameter and metric tracking.

### 2. Multi-Agent RAG Pipeline
- **ChromaDB Vector Store**: Persists built-in static SEO guidelines (Title rules, Meta descriptions, CTR optimization, etc.).
- **Dynamic Knowledge Fetching**: Automatically scrapes live SEO URLs to continuously update its knowledge base.
- **Agent Orchestration**: The `SEOAnalysisAgent` combines the ML model's raw score with the LangChain RAG pipeline to generate high-confidence, context-aware content recommendations using OpenRouter reasoning models (e.g., Nemotron).

### 3. Full-Stack Application
- **FastAPI Backend**: A highly performant API serving predictions, agent analysis, metrics, and workflow approvals.
- **React Frontend**: A premium, dark-themed "glassmorphism" UI delivering a rich dashboard, priority queues, and an interactive human-in-the-loop review process.

---

## 📂 Project Structure

```text
├── Agents/                 # Agent logic (model connection, RAG orchestration)
├── Datasets/               # Raw SEO metric datasets
├── MLModels/               # Machine Learning training pipelines & scripts
├── backend/                # FastAPI application & services
├── data/                   # Processed features & ChromaDB embeddings
├── feature_pipeline/       # Feature engineering & leakage audit scripts
├── frontend/               # React SPA (app.js, styles.css, index.html)
├── mlops/                  # Docker Compose, MLflow config, and drift monitoring
├── models/                 # Serialized best-performing ML models (.pkl)
├── rag/                    # Knowledge base builder & ChromaDB retriever
├── .env                    # API keys and environment variables
└── requirements.txt        # Python dependencies
```

## 📂 File Structure

```text
FlyRankMLoopS\
├── .env                          (existing — user fills API key + base_url)
├── requirements.txt              [NEW]
│
├── Datasets/                     (existing CSVs)
├── data/
│   ├── processed/                [NEW] cleaned features written here
│   └── embeddings/               [NEW] ChromaDB persisted here
│
├── MLModels/
│   ├── Regression1.py            (existing)
│   ├── TreeClassify.py           (existing — will be completed)
│   └── train_pipeline.py         [NEW] full training pipeline: RF + XGBoost + MLflow
│
├── feature_pipeline/
│   ├── _init_.py               [NEW]
│   ├── engineer.py               [NEW] feature engineering + derived features
│   └── leakage_audit.py          [NEW] leakage detection + time-based split
│
├── Agents/
│   ├── model.py                  (existing — FIXED: correct dotenv + openai init)
│   ├── RAG.py                    (existing — REBUILT: ChromaDB + proper retriever)
│   ├── chains.py                 (existing — FIXED: correct imports)
│   └── seo_agent.py              [NEW] single unified AI agent (SEO + content + decision)
│
├── rag/
│   ├── _init_.py               [NEW]
│   ├── knowledge_base.py         [NEW] SEO guidelines + brand knowledge embeddings
│   └── retriever.py              [NEW] ChromaDB retriever wrapper
│
├── backend/
│   ├── _init_.py               [NEW]
│   ├── main.py                   [NEW] FastAPI app with all endpoints
│   ├── schemas.py                [NEW] Pydantic request/response models
│   └── services/
│       ├── predict_service.py    [NEW] ML prediction service
│       ├── rag_service.py        [NEW] RAG query service
│       └── agent_service.py      [NEW] AI agent orchestration
│
├── mlops/
│   ├── mlflow_config.py          [NEW] MLflow experiment setup
│   ├── monitoring.py             [NEW] Evidently drift reports
│   └── docker-compose.yml        [NEW] full stack orchestration
│
├── frontend/
│   ├── index.html                [NEW] React SPA (single file, no build step needed)
│   ├── app.js                    [NEW] React via CDN
│   └── styles.css                [NEW] premium dark design
│
└── notebooks/
    └── exploration.ipynb         [NEW] optional EDA notebook
```


---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- (Optional) Docker

### 1. Clone & Install
```bash
git clone https://github.com/RohanNK86/SEO_Advanced_Automation.git
cd SEO_Advanced_Automation

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment variables
Create a `.env` file in the root directory and add your LLM credentials. (We use OpenRouter and Nvidia's Nemotron reasoning model by default).
```env
api_keys=your_openrouter_api_key_here
base_url=https://openrouter.ai/api/v1
llm_model=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

---

## 🚀 How to Run and Test the Platform

### Phase 1: Run the ML Pipeline
Process the data, audit for leakage, and train the predictive models.
```bash
# 1. Feature Engineering
python feature_pipeline/engineer.py

# 2. Leakage Audit
python feature_pipeline/leakage_audit.py

# 3. Train Classification and Regression Models
python MLModels/TreeClassify.py
python MLModels/train_pipeline.py
```
> *Check MLflow tracking using:* `mlflow ui --backend-store-uri sqlite:///mlflow.db`

### Phase 2: Start the Backend (FastAPI)
The backend handles the ML predictions and orchestrates the RAG agent.
```bash
python -m uvicorn backend.main:app --port 8001
```
> *Access the API Documentation:* `http://localhost:8001/docs`

### Phase 3: Start the Frontend (React Dashboard)
Serve the frontend dashboard using Python's built-in HTTP server.
```bash
python -m http.server 3000 --directory frontend
```
> *Access the Dashboard:* `http://localhost:3000`

---

## 🧪 Testing the Workflow

1. **Dashboard Overview**: Visit the frontend to view aggregated metrics (Pages Analyzed, Needs Refresh, Avg Score).
2. **Action Queue**: Review the Priority Action Queue table which sorts content based on the ML-predicted score.
3. **Agent Analysis**: Click on any page row to open the Analysis Modal. This will trigger:
   - A fast **ML prediction** for the refresh score and reason codes.
   - The **RAG Pipeline**, which constructs a query based on the page's metrics, fetches SEO rules from ChromaDB, and passes the context to the LLM.
   - The **LLM** returning a human-readable diagnosis and specific SEO improvements (Title, Meta Description, Content Tweaks).
4. **Human-in-the-Loop**: Use the "Approve Action" or "Reject Refresh" buttons to simulate an MLOps content review workflow.

---

*Built for advanced SEO automation and MLOps portfolio demonstration.*
