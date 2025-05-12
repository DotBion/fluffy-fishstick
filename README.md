# FinPulse: Stock Sentiment-Aware Price Forecasting + MLStack

FinPulse is a production-grade hybrid ML system combining market data with real-time financial sentiment from sources like news and Twitter to predict short-term stock price movements. It is fully automated using MLOps practices and deployed on Chameleon Cloud, targeting quantitative hedge funds and alt-data research teams.

## 1. Value Proposition

**Target Customers:**

* Quantitative research teams at hedge funds (e.g., Two Sigma, Citadel, Bloomberg)
* Personal investors

**Current Challenges:**

* Fragmented workflows
* Minimal monitoring and ad-hoc retraining

**Our Solution:**
FinPulse automates:

* Real-time data ingestion (Twitter, Vantage)
* Sentiment extraction (FinBERT)
* Price prediction (LSTM)
* Model serving (FastAPI)
* Monitoring (Prometheus, Grafana)
* Continuous retraining (Ray, MLflow)

**Business Metrics:**

* Accuracy: MAE, MAPE for sentiment classification
* Latency tracking
* Scheduled retraining and drift detection

## 2. Scale

**Data Scale:**

* Kaggle Tweet Dataset
* Daily OHLC & Volume from Vantage (10+ years, \~2500 data points per stock)

**Model Scale:**

* FinBERT (\~110M parameters) from ProsusAI
* LSTM for time-series and sentiment features

## Components Overview

| Component      | Technology Stack                |
| -------------- | ------------------------------- |
| Text Ingestion | Kaggle                          |
| Market Data    | Vantage                         |
| Model Training | LSTM, FinBERT                   |
| Serving        | Flask, Docker                   |
| Monitoring     | Prometheus, Grafana             |
| CI/CD          | GitHub Actions, Terraform, Helm |
| Persistence    | MinIO, Chameleon Volumes        |
| Infra-as-Code  | Terraform, Ansible, ArgoCD      |

## 3. Continuous X & DevOps

### CI/CD Workflow

End-to-end automation supporting continuous training, containerization, and deployment of LSTM models on Chameleon Cloud.

### Infrastructure as Code (IaC)

* Kubernetes (Terraform)
* Automated configurations (Ansible)
* Core services setup (ArgoCD, MLflow, MinIO, PostgreSQL)

### Terraform Provisioning

* Defined resources in `.tf` files
* Provisioned 3-node cluster
* IP management for deployment

### Ansible Configuration

* Kubernetes via Kubespray
* Networking, RBAC, Helm setup
* Full Kubernetes cluster automation

### Platform Services with ArgoCD + Helm

* MLflow, MinIO, PostgreSQL deployments
* Dynamic IP management
* Secret management

### Argo Workflows

* Continuous training and image building
* Model versioning and deployment
* Multi-environment deployments (Staging → Canary → Production)

## 4. Data Pipeline

### Offline Pipeline

### Sources:

* Kaggle datasets (financial news, historical prices)
* Alpha Vantage API (daily stock price data for AAPL, MSFT)
* CSVs or JSON files from `/mnt/block/raw/` (manual uploads or external sources)

### Processing:

* ETL pipelines defined in Airflow DAGs:

  * Fetch raw data from APIs or Kaggle
  * Clean, deduplicate, and reformat data using Python (pandas)
  * Add derived features (moving averages, percent changes)
* DAGs scheduled or manually triggered via Airflow UI

### Storage:

* MinIO (S3-compatible) used as the main data repository for:

  * Cleaned and versioned datasets (e.g., `processed_data/aapl_2025-05-11.csv`)
  * Raw Kaggle/API dumps (e.g., `raw_data/raw_kaggle_dump.csv`)
* Persistent block storage on Chameleon (`/mnt/block`) for:

  * Training artifacts (models, checkpoints)
  * DAG logs, requirements, container images if necessary
  * Backup or sync of raw/processed MinIO data

### Interactive Data Dashboard

* Streamlit dashboard for data inspection and drift trends (accessible at `/dashboard`)
* Users can explore tweet volume, sentiment polarity, and price correlation

### Offline Pipeline Overview & Flow:

### Node Setup:

* Conducted on node1 of Chameleon Cloud with floating IP for external access

### Dockerized Orchestration:

* Custom `docker-compose.yml` launching:

  * Airflow-webserver, Airflow-scheduler, Airflow-init containers
  * MinIO object store for persistent storage
* All services interconnected and managed via Docker

### Persistent Storage:

* 20GB persistent block storage volume mounted at `/mnt/block`
* Used to store intermediate and processed data
* Accessible by Airflow and Python containers

### Data Sources:

* Kaggle datasets stored locally
* Airflow DAG fetching updated stock data via Alpha Vantage API

### ETL via Airflow:

* Airflow DAGs fetch data (Kaggle/API), process it with Python operators, and upload cleaned data to MinIO
* Modular and reproducible DAGs scheduled via Airflow UI

### Data Repository:

* MinIO serves as central object store for raw and processed datasets

### EDA Dashboard:

* Dashboard developed for Exploratory Data Analysis
* Reads data from `/mnt/block/` or MinIO
* Provides visualizations, filters, and data quality summaries

## 5. Model Training

### Setup:

* Cleaned tweet dataset for FinBERT model
* Labeled tweets with sentiment scores using VADER
* Financial dataset appended with sentiment scores
* Target variables: next-day closing price, next-day sentiment prediction

### Models Used:

* FinBERT for sentiment prediction (`train_finbert.py`)
* LSTM for stock price prediction (`lstm_train.ipynb`, `lstm_train_pytorch.py`)
* VADER for dataset labeling (`tweets_2018_limited.csv`, `data_2018.csv`)

### Experiment Tracking:

* Hyperparameters tracked via MLflow hosted on CHI\@UC (`192.5.87.29`)
* Logged metrics: MAE, accuracy, latency
* Logged artifacts: model checkpoints, tokenizer, configuration (`lstm_model.pth`)

## 6. Model Serving

### API Endpoint:

* Hosted via Flask API
* LSTM API: `http://129.114.27.146:9090/predict`
* FinBERT API: `http://129.114.27.146:8080/predict`

### LSTM:

* Input: List of prices in JSON format
* Output: Predicted next-day price (Python List)

### FinBERT:

* Input: JSON with recent tweets and market indicators
* Output: Sentiment score and confidence as Python dictionary

### Model Optimizations:

* Throughput and inference time tests conducted with Prometheus and Grafana
* Evaluated model sizes across different architectures


## 7. Evaluation & Monitoring

**Offline Evaluation:**

* Metrics: MAE, MAPE, BLEU, accuracy
* Dataset split: 70/15/15
* MLflow tracking

**Load Testing:**

* Artillery simulations (1K+ concurrent requests)
* Stability tested up to 150 RPS

**Drift Monitoring:**

* Embedding-based input drift alerts
* Grafana visualizations

## 8. Chameleon Cloud Resources

| Resource     | Usage Purpose               |
| ------------ | --------------------------- |
| gpu\_a100    | FinBERT fine-tuning         |
| m1.large     | MLflow, Ray, ETL            |
| m1.medium    | FastAPI serving, monitoring |
| Floating IP  | Public API access           |
| 100GB Volume | Models, data, logs          |

## 9. Contributors

| Name              | Role                                                |
| ----------------- | --------------------------------------------------- |
| Aviraj Dongare    | CI/CD, Infra-as-Code                                |
| Ronit Gehani      | Data pipeline, Airflow, MinIO                       |
| Deeptanshu Lnu    | LSTM training, APIs, frontend, testing              |
| Nobodit Choudhury | Model Training, MLFlow                              |
| All Members       | Architecture, documentation, presentation           |

## Setup Instructions

**Prerequisites:**

* Chameleon account, SSH keypair, `clouds.yaml`

**Step-by-Step:**

Clone the Repo:

```bash
git clone https://github.com/<your-org>/mlops-project10.git
cd mlops-project10
```

Run Terraform:

```bash
cd tf/kvm
export TF_VAR_suffix=<netid>
export TF_VAR_key=id_rsa_chameleon
terraform init
terraform apply -auto-approve
```

Configure Kubernetes:

```bash
cd ../../ansible
ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml
ansible-playbook -i inventory.yml k8s/kubespray/cluster.yml
ansible-playbook -i inventory.yml post_k8s/post_k8s_configure.yml
```

Deploy Platform Services:

```bash
ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml
```

Build Base Images:

```bash
ansible-playbook -i inventory.yml argocd/workflow_build_init.yml
```

Deploy Models:

```bash
ansible-playbook -i inventory.yml argocd/argocd_add_staging.yml
ansible-playbook -i inventory.yml argocd/argocd_add_canary.yml
ansible-playbook -i inventory.yml argocd/argocd_add_prod.yml
```

Trigger Training:

```bash
ssh -L 2746:127.0.0.1:2746 -i ~/.ssh/id_rsa_chameleon cc@129.114.26.118
```

Access dashboards:

* MLflow: `http://129.114.26.118:8000`
* MinIO: `http://129.114.26.118:9001`
