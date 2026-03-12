Support Ticket Intelligence System

ML-powered system that analyzes customer support tickets to automatically detect issue clusters, identify emerging trends, and surface anomalies — without using paid APIs.

Built using Python + scikit-learn + React + Vite.




System Architecture
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
Features
Automatic Issue Detection

Clusters support tickets into issue groups using machine learning.

Trend Detection

Identifies whether issues are:

increasing

decreasing

stable

using sliding window analysis.

Anomaly Detection

Detects unusual spikes in support tickets using statistical thresholds.

Interactive Dashboard

React dashboard provides:

issue cluster cards

trend charts

anomaly detection view

cluster deep-dive panel

Ticket Simulation

Generate synthetic tickets to test the pipeline.

Real-Time Analysis Refresh

Re-run the ML pipeline after new tickets are added.

How It Works
1. Text Preprocessing

Ticket fields are cleaned and combined:

Subject + Ticket Type + Product + Description

Processing steps:

remove emails, URLs, phone numbers

normalize whitespace

lowercase text

replace template variables

Subject is double weighted for stronger signal.

Feature Extraction

Tickets are converted into vectors using TF-IDF.

max_features = 5000
ngram_range = (1,2)
min_df = 3
max_df = 0.85
sublinear_tf = True

This captures phrases like:

"wifi disconnect"
"battery issue"
"network timeout"
Clustering Pipeline
TF-IDF → SVD → Normalization → KMeans

Steps:

1️⃣ TF-IDF vectorization
2️⃣ Dimensionality reduction using TruncatedSVD (LSA)
3️⃣ L2 normalization
4️⃣ Optimal k selection using silhouette score
5️⃣ MiniBatch KMeans clustering

Each ticket is assigned to an issue cluster.

Cluster Labeling

Clusters are labeled using:

most common subject

top TF-IDF terms

ticket type

most common affected products

Example:

Cluster: WiFi disconnecting
Top Terms: wifi, signal, drop
Products: Router Pro, Router X
Trend Detection

Trend classification uses sliding windows:

Previous Window (3 months)
Current Window (3 months)
pct_change = ((curr - prev) / prev) * 100

Classification:

Change	Trend

+15% | Increasing |
< −15% | Decreasing |
otherwise | Stable |

Anomaly Detection

Monthly cluster volumes are analyzed.

threshold = mean + 2 × std

Months exceeding the threshold are flagged as anomalies.

These represent possible:

outages

incidents

regressions

Severity Scoring

Clusters are ranked by severity:

severity = ticket_count × (1 + pct_change/100)

This prioritizes issues that are both large and growing quickly.

Tech Stack
Layer	Technology
ML	scikit-learn
Backend	Python
Frontend	React + Vite
Charts	Recharts
Data	Pandas
Running the Project
Backend

Install dependencies:

pip install pandas numpy scikit-learn

Start server:

cd backend
python server.py

Backend runs at:

http://localhost:8080
Frontend

Install dependencies:

cd frontend
npm install

Run dev server:

npm run dev

Frontend runs at:

http://localhost:5173
API Endpoints
GET  /api/analysis
GET  /api/cluster/:id
GET  /api/health

POST /api/ingest
POST /api/simulate
POST /api/refresh
Project Structure
support-ticket-intelligence/
│
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
Design Decisions
Decision	Reason
TF-IDF over embeddings	no paid APIs required
KMeans clustering	stable and interpretable
SVD dimensionality reduction	faster clustering
Sliding window trends	simple + interpretable
CSV dataset	lightweight storage
Future Improvements

Possible upgrades:

Sentence-BERT embeddings for better semantic clustering

WebSocket streaming updates

PostgreSQL + pgvector storage

automated Slack/email alerting

root-cause correlation with releases

customer impact scoring

License

MIT License
