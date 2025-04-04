# 📈 Stock Sentiment-Aware Price Forecasting using LLM + MLStack

A hybrid machine learning system that estimates future stock prices by combining real-time textual financial sentiment (news, Twitter, etc.) and traditional market indicators. The system leverages LLMs like **FinBERT**, custom embeddings, and regression-based predictors, and is built with a full MLOps pipeline on **Chameleon Cloud**.

---

## 🧠 Value Proposition

Our project targets quantitative hedge funds and alternative data research teams at firms like Two Sigma, Citadel Securities, or Bloomberg. These institutions constantly seek scalable ways to integrate alternative signals like public sentiment into trading models. FinPulse provides a production-ready, automated pipeline that processes daily financial news and market data to predict next-day stock price movement.

Unlike fragmented research workflows or experimental scripts, FinPulse enables end-to-end automation of the ML lifecycle: data ingestion, feature engineering, model training, serving, monitoring, and retraining. It significantly reduces the latency between research insight and model deployment—empowering firms to capitalize on market sentiment quickly and reliably.

- **Current Status Quo**: Today, data scientists in hedge funds and financial analytics firms work in fragmented environments:
Sentiment analysis and price modeling are often done independently.
Transitioning from research to deployment is manual and time-consuming.
Retraining schedules are ad hoc, leading to model staleness.
Monitoring and observability are limited—data drifts or system failures may go unnoticed.

- **Our Improvement**: FinPulse bridges the gap between alpha discovery and operationalization, helping funds move faster from signal to trade while maintaining transparency and robustness.
- **Business Metric**: Accuracy of predictions, latency for inference, model Freshness

---

## 👥 Contributors

| Name               | Role                                        | Commits |
|--------------------|---------------------------------------------|---------|
| All team members   | Design, Infra setup, Documentation          | [Link](#) |
| Ronit Gehani       | Data pipeline, MLflow, FSDP                 | [Link](#) |
| Deeptanshu Lnu     | Training pipeline, FinBERT optimization     | [Link](#) |
| Aviraj Dongare     | CI/CD, GitHub Actions, Infra-as-Code        | [Link](#) |
| Nobodit Choudhury  | FastAPI backend, serving infra              | [Link](#) |


---

## 🗂 System Overview

![architecture-block-v2](https://github.com/user-attachments/assets/0690acc0-020f-430c-aff8-65fd312ad57d)


### Components

- 📰 **Text Sources**: Financial PhraseBank, Twitter, News APIs
- 📊 **Market Data**: YFinance API
- 🔤 **NLP**: FinBERT fine-tuned for sentiment
- 📈 **Prediction Model**: Logistic/XGBoost with embeddings
- 📦 **Pipeline**: Spark + Airflow for ETL
- 🧪 **Experiment Tracking**: MLflow + Ray
- 🚀 **Deployment**: Docker + FastAPI
- 🔍 **Monitoring**: Grafana + Loki + Prometheus

---

## 📦 Datasets & Models

| Resource             | Origin/Creation                    | License |
|----------------------|-------------------------------------|---------|
| Financial PhraseBank | Annotated financial text dataset    | Open Use |
| Twitter API          | Live tweets                        | Academic |
| YFinance             | Stock price data                   | Public API |
| FinBERT              | Pretrained NLP model               | Apache 2.0 |

---

## ☁️ Chameleon Cloud Infrastructure

| Resource       | Usage Purpose                          |
|----------------|-----------------------------------------|
| `gpu_a100`     | FinBERT fine-tuning, embedding gen      |
| `m1.large`     | Ray head node, MLflow, Airflow          |
| `m1.medium`    | API server, Monitoring (Grafana/Loki)   |
| Floating IPs   | FastAPI + Dashboard access              |
| 100GB Volume   | Persist models, embeddings, logs        |

---

## 🛠️ Project Design

### 🧠 Model Training

- Fine-tuning FinBERT + sentiment embeddings
- Market embeddings from structured data
- Combined model via Logistic Regression or XGBoost
- Tracked in **MLflow**, scheduled via **Ray**
- ✅ **Difficulty**: Distributed training (Ray Train), Ray Tune HPO

### 🚀 Model Serving & Monitoring

- FastAPI Docker app for `/predict`
- Prometheus + Grafana dashboards
- ✅ **Difficulty**: Compare CPU/GPU inference + live drift monitoring

### 🔄 Data Pipeline

- **Offline**: Twitter/YFinance → ETL → Postgres
- **Online**: Simulated API calls for real-time update
- ✅ **Difficulty**: Interactive data quality dashboard

### 🔁 Continuous Integration

- CI/CD via GitHub Actions:
  - ETL → Train → Evaluate → Docker Build → Helm Deploy
- Staging/Canary/Prod with Helm
  

---

## 📊 Evaluation Plan

- Offline evaluation: BLEU, MAPE, MAE on held-out test set
- Online evaluation: User simulation tests
- Drift detection: Based on embeddings, market data change
- Feedback loop: Real-time prediction storage for retraining

---

## 📎 License & Usage

This project is intended for academic and research purposes only. All third-party models and datasets comply with their respective licenses.

---

## ✨ Live Demo / API Endpoint

> Coming soon: [https://your-chameleon-ip](https://your-chameleon-ip)

---
