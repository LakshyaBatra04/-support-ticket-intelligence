"""
ML Engine for Support Ticket Intelligence System
=================================================
Uses TF-IDF + K-Means clustering for issue detection,
and sliding window analysis for trend detection.
All processing is done locally with scikit-learn — no paid APIs.
"""

import re
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


class TicketAnalyzer:
    """
    Core ML pipeline:
    1. Text preprocessing & cleaning
    2. TF-IDF vectorization of ticket text
    3. Dimensionality reduction via TruncatedSVD (LSA)
    4. K-Means clustering to find issue groups
    5. Cluster labeling from top TF-IDF terms
    6. Sliding window trend detection
    """

    def __init__(self):
        self.vectorizer = None
        self.svd = None
        self.kmeans = None
        self.cluster_labels = {}
        self.tickets_df = None
        self.cluster_results = None
        self.trend_results = None
        self.n_clusters = 12  # will be tuned

    # ── Text Preprocessing ──────────────────────────────────────────────

    @staticmethod
    def clean_text(text):
        """Clean ticket text: remove template vars, noise, normalize."""
        if not isinstance(text, str):
            return ""
        # Replace template variables like {product_purchased}
        text = re.sub(r'\{[^}]+\}', 'PRODUCT', text)
        # Remove emails, phone numbers, zip codes, URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'1-\d{3}-\d{3}-\d{4}', '', text)
        text = re.sub(r'\b\d{5}\b', '', text)
        text = re.sub(r'https?://\S+', '', text)
        # Remove special chars but keep meaningful punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def preprocess_tickets(self, df):
        """Combine subject + description + type for richer text signal."""
        df = df.copy()
        df['clean_description'] = df['Ticket Description'].apply(self.clean_text)
        df['clean_subject'] = df['Ticket Subject'].apply(
            lambda x: x.lower().strip() if isinstance(x, str) else ''
        )
        # Combine fields — subject is repeated for higher weight
        df['combined_text'] = (
            df['clean_subject'] + ' ' +
            df['clean_subject'] + ' ' +  # double-weight subject
            df['Ticket Type'].fillna('').str.lower() + ' ' +
            df['Product Purchased'].fillna('').str.lower() + ' ' +
            df['clean_description']
        )
        # Parse dates
        df['date'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
        df['year_month'] = df['date'].dt.to_period('M')
        self.tickets_df = df
        return df

    # ── Clustering Pipeline ─────────────────────────────────────────────

    def find_optimal_clusters(self, X, min_k=8, max_k=20):
        """Use silhouette score to find best k for K-Means."""
        best_score = -1
        best_k = 12
        for k in range(min_k, max_k + 1, 2):
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels, sample_size=min(2000, X.shape[0]))
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    def cluster_tickets(self):
        """
        Full clustering pipeline:
        TF-IDF → SVD (LSA) → K-Means → Label extraction
        """
        df = self.tickets_df

        # TF-IDF with bigrams for better phrase capture
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85,
            stop_words='english',
            sublinear_tf=True  # log normalization
        )
        tfidf_matrix = self.vectorizer.fit_transform(df['combined_text'])

        # Dimensionality reduction with LSA (Latent Semantic Analysis)
        n_components = min(100, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = self.svd.fit_transform(tfidf_matrix)
        X_reduced = normalize(X_reduced)  # L2 normalize for better clustering

        # Find optimal cluster count
        self.n_clusters = self.find_optimal_clusters(X_reduced)

        # K-Means clustering
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1024,
            n_init=10
        )
        df['cluster'] = self.kmeans.fit_predict(X_reduced)

        # Store reduced features for later use
        df['_features'] = list(X_reduced)

        # Generate cluster labels from top TF-IDF terms
        self._label_clusters(tfidf_matrix, df)

        self.tickets_df = df
        return df

    def _label_clusters(self, tfidf_matrix, df):
        """
        Extract meaningful labels for each cluster using:
        1. Top TF-IDF terms per cluster centroid
        2. Most common ticket subjects in cluster
        """
        feature_names = self.vectorizer.get_feature_names_out()

        for cluster_id in range(self.n_clusters):
            mask = df['cluster'] == cluster_id
            cluster_tfidf = tfidf_matrix[mask.values]

            # Mean TF-IDF vector for this cluster
            mean_tfidf = cluster_tfidf.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-8:][::-1]
            top_terms = [feature_names[i] for i in top_indices]

            # Most common subjects in this cluster
            subjects = df.loc[mask, 'Ticket Subject'].value_counts()
            top_subject = subjects.index[0] if len(subjects) > 0 else 'Unknown'

            # Most common ticket type
            types = df.loc[mask, 'Ticket Type'].value_counts()
            top_type = types.index[0] if len(types) > 0 else ''

            # Most common products
            products = df.loc[mask, 'Product Purchased'].value_counts()
            top_products = list(products.index[:3])

            self.cluster_labels[cluster_id] = {
                'name': top_subject,
                'top_terms': top_terms,
                'top_type': top_type,
                'top_products': top_products,
                'subject_distribution': subjects.head(5).to_dict(),
                'count': int(mask.sum())
            }

    # ── Trend Detection ─────────────────────────────────────────────────

    def detect_trends(self, window_months=3):
        """
        Sliding window trend detection:
        - Compare recent window vs previous window
        - Calculate percentage change and classify trend
        - Also compute per-month time series for charting
        """
        df = self.tickets_df
        trends = {}

        # Get date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        mid_date = max_date - pd.DateOffset(months=window_months)

        for cluster_id in range(self.n_clusters):
            cluster_df = df[df['cluster'] == cluster_id]

            # Previous vs current window counts
            prev_count = len(cluster_df[(cluster_df['date'] >= mid_date - pd.DateOffset(months=window_months)) & (cluster_df['date'] < mid_date)])
            curr_count = len(cluster_df[cluster_df['date'] >= mid_date])

            # Percentage change
            if prev_count > 0:
                pct_change = ((curr_count - prev_count) / prev_count) * 100
            else:
                pct_change = 100 if curr_count > 0 else 0

            # Classify trend
            if pct_change > 15:
                trend = 'increasing'
            elif pct_change < -15:
                trend = 'decreasing'
            else:
                trend = 'stable'

            # Monthly time series for sparkline/chart
            monthly = cluster_df.groupby('year_month').size()
            time_series = []
            periods = pd.period_range(
                start=min_date.to_period('M'),
                end=max_date.to_period('M'),
                freq='M'
            )
            for period in periods:
                count = monthly.get(period, 0)
                time_series.append({
                    'month': str(period),
                    'count': int(count)
                })

            # Severity score: combines volume + trend
            severity = (curr_count * (1 + max(0, pct_change) / 100))

            trends[cluster_id] = {
                'prev_window': int(prev_count),
                'curr_window': int(curr_count),
                'pct_change': round(pct_change, 1),
                'trend': trend,
                'time_series': time_series,
                'severity': round(severity, 1)
            }

        self.trend_results = trends
        return trends

    # ── Additional Analytics ────────────────────────────────────────────

    def get_priority_breakdown(self):
        """Priority distribution per cluster."""
        df = self.tickets_df
        result = {}
        for cluster_id in range(self.n_clusters):
            cluster_df = df[df['cluster'] == cluster_id]
            priority_counts = cluster_df['Ticket Priority'].value_counts().to_dict()
            result[cluster_id] = priority_counts
        return result

    def get_resolution_stats(self):
        """Resolution time and satisfaction stats per cluster."""
        df = self.tickets_df
        result = {}
        for cluster_id in range(self.n_clusters):
            cluster_df = df[df['cluster'] == cluster_id]
            # Parse resolution time
            res_times = pd.to_numeric(
                cluster_df['Time to Resolution'].str.extract(r'(\d+)')[0],
                errors='coerce'
            )
            sat_ratings = pd.to_numeric(
                cluster_df['Customer Satisfaction Rating'], errors='coerce'
            )
            result[cluster_id] = {
                'avg_resolution_hours': round(res_times.mean(), 1) if not res_times.isna().all() else None,
                'avg_satisfaction': round(sat_ratings.mean(), 2) if not sat_ratings.isna().all() else None,
                'open_tickets': int((cluster_df['Ticket Status'] == 'Open').sum()),
                'pending_tickets': int((cluster_df['Ticket Status'] == 'Pending Customer Response').sum()),
                'closed_tickets': int((cluster_df['Ticket Status'] == 'Closed').sum()),
            }
        return result

    def get_channel_distribution(self):
        """Which channels tickets come from per cluster."""
        df = self.tickets_df
        result = {}
        for cluster_id in range(self.n_clusters):
            cluster_df = df[df['cluster'] == cluster_id]
            result[cluster_id] = cluster_df['Ticket Channel'].value_counts().to_dict()
        return result

    # ── Anomaly Detection ───────────────────────────────────────────────

    def detect_anomalies(self):
        """
        Detect anomalous spikes: months where a cluster's count
        exceeds 2 standard deviations above its mean.
        """
        df = self.tickets_df
        anomalies = []

        for cluster_id in range(self.n_clusters):
            cluster_df = df[df['cluster'] == cluster_id]
            monthly = cluster_df.groupby('year_month').size()

            if len(monthly) < 3:
                continue

            mean_count = monthly.mean()
            std_count = monthly.std()
            threshold = mean_count + 2 * std_count

            for period, count in monthly.items():
                if count > threshold:
                    anomalies.append({
                        'cluster_id': int(cluster_id),
                        'cluster_name': self.cluster_labels[cluster_id]['name'],
                        'month': str(period),
                        'count': int(count),
                        'expected': round(mean_count, 1),
                        'threshold': round(threshold, 1)
                    })

        return sorted(anomalies, key=lambda x: x['count'], reverse=True)

    # ── Export Full Results ──────────────────────────────────────────────

    def get_all_tickets_sorted(self, cluster_id):
        """Get ALL tickets for a cluster, sorted by date (most recent first)."""
        df = self.tickets_df
        cluster_df = df[df['cluster'] == cluster_id].sort_values('date', ascending=False)

        tickets = []
        for _, row in cluster_df.iterrows():
            desc = row['Ticket Description']
            if len(desc) > 200:
                desc = desc[:200] + '...'
            tickets.append({
                'ticket_id': int(row['Ticket ID']),
                'subject': row['Ticket Subject'],
                'description': desc,
                'product': row['Product Purchased'],
                'priority': row['Ticket Priority'],
                'date': str(row['date'].date()) if pd.notna(row['date']) else '',
                'status': row['Ticket Status'],
                'type': row['Ticket Type'],
                'channel': row['Ticket Channel'],
            })

        return tickets

    def get_full_results(self):
        """Compile all analysis into a single JSON-serializable dict."""
        clusters = []
        priority_breakdown = self.get_priority_breakdown()
        resolution_stats = self.get_resolution_stats()
        channel_dist = self.get_channel_distribution()
        trends = self.trend_results or self.detect_trends()
        anomalies = self.detect_anomalies()

        for cluster_id in range(self.n_clusters):
            label = self.cluster_labels[cluster_id]
            trend = trends[cluster_id]
            examples = self.get_all_tickets_sorted(cluster_id)

            clusters.append({
                'id': cluster_id,
                'name': label['name'],
                'count': label['count'],
                'top_terms': label['top_terms'],
                'top_type': label['top_type'],
                'top_products': label['top_products'],
                'subject_distribution': label['subject_distribution'],
                'trend': trend['trend'],
                'pct_change': trend['pct_change'],
                'prev_window': trend['prev_window'],
                'curr_window': trend['curr_window'],
                'severity': trend['severity'],
                'time_series': trend['time_series'],
                'priority': priority_breakdown[cluster_id],
                'resolution': resolution_stats[cluster_id],
                'channels': channel_dist[cluster_id],
                'tickets': examples,
            })

        # Sort by severity (most critical first)
        clusters.sort(key=lambda c: c['severity'], reverse=True)

        # Global stats
        df = self.tickets_df
        global_stats = {
            'total_tickets': len(df),
            'date_range': {
                'start': str(df['date'].min().date()),
                'end': str(df['date'].max().date()),
            },
            'total_clusters': self.n_clusters,
            'increasing_issues': sum(1 for t in trends.values() if t['trend'] == 'increasing'),
            'decreasing_issues': sum(1 for t in trends.values() if t['trend'] == 'decreasing'),
            'stable_issues': sum(1 for t in trends.values() if t['trend'] == 'stable'),
            'avg_satisfaction': round(pd.to_numeric(df['Customer Satisfaction Rating'], errors='coerce').mean(), 2),
            'critical_tickets': int((df['Ticket Priority'] == 'Critical').sum()),
            'open_tickets': int((df['Ticket Status'] == 'Open').sum()),
            'type_distribution': df['Ticket Type'].value_counts().to_dict(),
            'priority_distribution': df['Ticket Priority'].value_counts().to_dict(),
            'monthly_volume': [
                {'month': str(p), 'count': int(c)}
                for p, c in df.groupby('year_month').size().items()
            ],
        }

        return {
            'clusters': clusters,
            'global_stats': global_stats,
            'anomalies': anomalies,
        }


def run_analysis(csv_path):
    """Run the full analysis pipeline and return JSON results."""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tickets")

    analyzer = TicketAnalyzer()

    print("Preprocessing tickets...")
    analyzer.preprocess_tickets(df)

    print("Running TF-IDF + K-Means clustering...")
    analyzer.cluster_tickets()
    print(f"Found {analyzer.n_clusters} clusters")

    print("Detecting trends...")
    analyzer.detect_trends()

    print("Compiling results...")
    results = analyzer.get_full_results()

    return results, analyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run support ticket ML analysis")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the customer_support_tickets.csv dataset",
    )
    parser.add_argument(
        "--output",
        default="analysis_results.json",
        help="Output JSON file path (default: analysis_results.json)",
    )

    args = parser.parse_args()

    results, analyzer = run_analysis(args.input)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nAnalysis complete!")
    print(f"Clusters: {results['global_stats']['total_clusters']}")
    print(f"Increasing issues: {results['global_stats']['increasing_issues']}")
    print(f"Anomalies detected: {len(results['anomalies'])}")