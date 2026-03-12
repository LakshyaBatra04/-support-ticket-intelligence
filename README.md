Your README is already good, but for **GitHub it should be structured more like a professional open-source project page** so that recruiters and developers immediately understand:

* what the project does
* how to run it
* what the architecture looks like
* what makes it interesting technically

Below is a **cleaned, GitHub-ready README** with better formatting, sections, badges, and developer-friendly structure.

---

# Support Ticket Intelligence System

ML-powered system that **automatically analyzes customer support tickets** to detect issue clusters, identify emerging trends, and surface anomalies — **without relying on paid APIs**.

Built using **Python + scikit-learn + React + Vite**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![React](https://img.shields.io/badge/React-Dashboard-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

# Overview

Customer support teams often receive thousands of tickets, making it difficult to:

* identify recurring issues
* detect spikes in incidents
* understand product health
* prioritize fixes

This system applies **machine learning and statistical analysis** to automatically discover patterns in support tickets.

The platform provides a **dashboard + ML pipeline** that surfaces:

* issue clusters
* emerging trends
* anomaly spikes
* severity rankings

---

# System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│   React Dashboard                                            │
│   ├── Issue Cluster Cards                                    │
│   ├── Trend Analysis Table                                   │
│   ├── Anomaly Detection View                                 │
│   ├── Cluster Detail Panel                                   │
│   └── Ticket Ingestion UI                                    │
├─────────────────────────────────────────────────────────────┤
│                        REST API                              │
│                        server.py                             │
│                                                             │
│   GET  /api/analysis        → Full ML analysis               │
│   GET  /api/cluster/:id     → Cluster details                │
│   POST /api/ingest          → Add ticket                     │
│   POST /api/simulate        → Generate synthetic tickets     │
│   POST /api/refresh         → Re-run ML pipeline             │
├─────────────────────────────────────────────────────────────┤
│                       ML Engine                              │
│                       ml_engine.py                           │
│                                                             │
│   Text Cleaning → TF-IDF → SVD → KMeans → Trends → Anomaly   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                         Data                                 │
│                         CSV Dataset                          │
└─────────────────────────────────────────────────────────────┘
```

---

# Features

## Automatic Issue Detection

Clusters support tickets into **issue groups** using machine learning.

Example clusters:

* WiFi disconnecting
* Battery draining
* Login failures
* Network timeouts

---

## Trend Detection

Detects whether an issue is:

| Change         | Trend      |
| -------------- | ---------- |
| +15% or higher | Increasing |
| −15% or lower  | Decreasing |
| otherwise      | Stable     |

Uses **sliding time windows** to detect changes.

---

## Anomaly Detection

Flags **unusual spikes** in ticket volume using statistical thresholds.

```
threshold = mean + 2 × std
```

Useful for detecting:

* production incidents
* outages
* regressions after releases

---

## Interactive Dashboard

React dashboard provides:

* Issue cluster cards
* Trend analysis table
* Anomaly detection panel
* Cluster deep-dive view
* Ticket ingestion UI

---

## Ticket Simulation

Generate **synthetic tickets** to test the ML pipeline.

Useful for demos and stress testing.

---

## Real-Time Pipeline Refresh

After new tickets are ingested the ML pipeline can be **re-run instantly**.

---

# How the ML Pipeline Works

## 1. Text Preprocessing

Ticket fields are combined:

```
Subject + Ticket Type + Product + Description
```

Cleaning steps:

* remove emails
* remove URLs
* remove phone numbers
* normalize whitespace
* lowercase text
* replace template variables

Subject is **double weighted** to improve clustering signal.

---

# Feature Extraction

Tickets are converted into vectors using **TF-IDF**.

Parameters:

```
max_features = 5000
ngram_range = (1,2)
min_df = 3
max_df = 0.85
sublinear_tf = True
```

This captures phrases like:

* wifi disconnect
* battery issue
* network timeout
* login error

---

# Clustering Pipeline

```
TF-IDF → SVD → Normalization → KMeans
```

Steps:

1. TF-IDF vectorization
2. Dimensionality reduction using **TruncatedSVD (LSA)**
3. L2 normalization
4. Optimal cluster count using **silhouette score**
5. **MiniBatch KMeans** clustering

Each ticket is assigned to an **issue cluster**.

---

# Cluster Labeling

Clusters are labeled using:

* most common subject
* top TF-IDF terms
* ticket type
* most affected products

Example:

```
Cluster: WiFi Disconnecting

Top Terms:
wifi, signal, drop

Products:
Router Pro
Router X
```

---

# Trend Detection

Trend classification uses sliding windows:

```
Previous Window: 3 months
Current Window: 3 months

pct_change = ((curr - prev) / prev) × 100
```

This highlights **emerging issues early**.

---

# Anomaly Detection

Monthly cluster volumes are analyzed.

```
threshold = mean + 2 × std
```

Months exceeding this threshold are flagged as anomalies.

Possible causes:

* infrastructure outage
* software regression
* product defect

---

# Severity Scoring

Clusters are ranked by severity:

```
severity = ticket_count × (1 + pct_change / 100)
```

This prioritizes issues that are **both large and growing quickly**.

---

# Tech Stack

| Layer    | Technology   |
| -------- | ------------ |
| ML       | scikit-learn |
| Backend  | Python       |
| Frontend | React + Vite |
| Charts   | Recharts     |
| Data     | Pandas       |

---

# Running the Project

## Backend

Install dependencies:

```
pip install pandas numpy scikit-learn
```

Run the backend:

```
cd backend
python server.py
```

Backend runs at:

```
http://localhost:8080
```

---

## Frontend

Install dependencies:

```
cd frontend
npm install
```

Run development server:

```
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# API Endpoints

```
GET  /api/analysis
GET  /api/cluster/:id
GET  /api/health

POST /api/ingest
POST /api/simulate
POST /api/refresh
```

---

# Project Structure

```
support-ticket-intelligence/

├── backend/
│   ├── ml_engine.py
│   └── server.py
│
├── frontend/
│   ├── src/
│   │   └── support_intelligence.jsx
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   └── customer_support_tickets.csv
│
├── .gitignore
└── README.md
```

---

# Design Decisions

| Decision                     | Reason                   |
| ---------------------------- | ------------------------ |
| TF-IDF instead of embeddings | avoids paid APIs         |
| KMeans clustering            | stable and interpretable |
| SVD dimensionality reduction | faster clustering        |
| Sliding window trends        | simple and explainable   |
| CSV dataset                  | lightweight storage      |

---

# Future Improvements

Potential upgrades:

* Sentence-BERT embeddings for semantic clustering
* WebSocket streaming updates
* PostgreSQL + pgvector storage
* automated Slack/email alerts
* root-cause correlation with releases
* customer impact scoring

---

