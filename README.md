# Support Ticket Issue Intelligence System

A full-stack system that analyzes customer support tickets, automatically clusters
related issues, detects emerging trends, and surfaces actionable insights via a
real-time dashboard powered by Claude AI.

---

## Architecture Overview

```
CSV Dataset (8,469 tickets)
        │
        ▼
┌─────────────────────┐
│  backend/analyze.py │  ← Python ingestion & analysis
│                     │
│  1. Load & parse    │
│  2. Cluster by      │
│     subject/text    │
│  3. Trend detection │
│  4. Output JSON     │
└────────┬────────────┘
         │  analysis.json
         ▼
┌──────────────────────────┐
│  React Dashboard         │  ← frontend/support_intelligence.jsx
│                          │
│  • Cluster list sidebar  │
│  • Trend indicators      │
│  • Priority breakdowns   │
│  • Example tickets       │
│  • AI analysis (Claude)  │
│  • Ticket simulation     │
└──────────────────────────┘
```

---

## How Tickets Are Processed

1. **Ingestion**: The CSV is read and parsed. Only tickets with a `First Response Time`
   timestamp are used (5,650 of 8,469 records). This field anchors the ticket to a
   real processing moment.

2. **Field selection**: We use `Ticket Subject`, `Ticket Description`, `Ticket Priority`,
   `Ticket Type`, `Product Purchased`, and `First Response Time`. The subject is the
   most consistent signal for grouping; the description provides human-readable examples.

3. **Cleaning**: Ticket descriptions contain template artifacts (`{product_purchased}`,
   HTML fragments, etc.) — these are handled gracefully at display time by trimming to
   the first meaningful line and capping at 150 characters.

---

## How Issues Are Detected

**Approach: Taxonomy-based clustering**

The dataset's `Ticket Subject` field provides 16 pre-labeled issue categories
(e.g. "Network problem", "Data loss", "Payment issue"). Rather than applying
unsupervised clustering (k-means, DBSCAN) to raw text — which would reproduce
these same categories noisily — we use the labels directly as cluster identifiers.

This is a deliberate design choice: the dataset already encodes the taxonomy;
re-deriving it adds complexity without accuracy. In production with unlabeled data,
the approach would shift to TF-IDF + cosine similarity clustering or embedding-based
grouping (e.g. sentence-transformers + HDBSCAN).

Each cluster surfaces:
- Total ticket count
- Priority distribution (Critical / High / Medium / Low)
- Ticket type distribution (Technical issue / Billing inquiry / etc.)
- Representative example tickets

---

## How Trends Are Identified

**Method: Chronological half-split comparison**

All timestamped tickets are sorted by `First Response Time` and split into two
equal halves (earlier 50% vs later 50%). For each cluster:

```
prev_count = tickets in cluster that fall in first half
curr_count = tickets in cluster that fall in second half
delta      = (curr - prev) / prev

trend = "increasing" if delta > +10%
      = "decreasing" if delta < -10%
      = "stable" otherwise
```

**Why this approach over calendar months?**
The dataset spans only two calendar months (May–June 2023), making month-over-month
comparisons trivially show all clusters as "increasing" (since June has ~20x more
data than May in this dataset). The half-split is dataset-adaptive and produces
meaningful, comparable signal regardless of time distribution.

**Results from this dataset:**
| Cluster | Trend | Change |
|---|---|---|
| Data Loss | ↑ Increasing | +24% |
| Payment Issue | ↑ Increasing | +16% |
| Delivery Problem | ↑ Increasing | +15% |
| Installation Support | ↓ Decreasing | -11% |
| All others | → Stable | <10% delta |

---

## Dashboard Features

- **Cluster sidebar**: All 16 issue clusters ranked by volume with mini priority bars
- **Filter tabs**: View all / rising / falling / stable clusters
- **Trend badges**: Color-coded indicators with percentage change
- **Detail panel**: Clicking a cluster shows priority breakdown, ticket type distribution,
  and 3 real example tickets with priority tags
- **AI Analysis**: "Generate AI Analysis" calls Claude API to produce root cause
  hypotheses, customer impact assessment, and recommended actions for any cluster
- **Ticket Injection**: Simulate new incoming tickets to see how clusters update in
  real time (demonstrates the "update results when new tickets arrive" requirement)
- **Emerging Issues Banner**: Persistent alert bar when any cluster is trending up

---

## Design Decisions & Tradeoffs

| Decision | Rationale | Tradeoff |
|---|---|---|
| Taxonomy-based clustering | Labels already exist; highly accurate | Would need NLP for unlabeled data |
| Half-split trend detection | Dataset-adaptive; works with any time distribution | Less interpretable than calendar periods |
| Frontend-only architecture | No server needed; easy to run | Analysis runs at build time, not streaming |
| Claude AI analysis on-demand | Avoids pre-computing expensive AI calls | Small latency per click (~1-2s) |
| In-memory ticket simulation | Demonstrates update requirement simply | Resets on page reload (no persistence) |

---

## What I Would Improve With More Time

1. **Embedding-based clustering** — Use `sentence-transformers` to embed ticket
   descriptions, then HDBSCAN to find clusters that cross category boundaries
   (e.g. "network + setup" issues often co-occur).

2. **Sliding window trend detection** — Replace the half-split with a proper
   time-series approach: compute a 7-day rolling count per cluster and flag when
   it exceeds 2 standard deviations above the rolling mean.

3. **Persistent ticket injection** — Back the simulation with SQLite so injected
   tickets survive page reloads and accumulate over a session.

4. **Webhook / streaming ingestion** — Expose a POST `/ticket` endpoint that
   re-runs analysis incrementally rather than batch-processing the whole dataset.

5. **Cross-cluster correlation** — Detect when multiple clusters spike simultaneously
   (e.g. "Network problem" + "Account access" both rising may indicate an outage).

6. **Auto-generated cluster names** — When working with unlabeled data, use Claude
   to generate a descriptive name for each NLP-derived cluster automatically.

---

## Running Locally

```bash
# Backend — produce analysis JSON
cd backend
python3 analyze.py --input tickets.csv --output analysis.json

# Frontend — open the React dashboard
# Load support_intelligence.jsx in Claude.ai or any React sandbox
```

**Requirements**: Python 3.8+, standard library only (no pip installs needed).

---

## Dataset

Customer Support Ticket Dataset (Kaggle)  
https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset  
8,469 tickets · 5,650 with timestamps used
