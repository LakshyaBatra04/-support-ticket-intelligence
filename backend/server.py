import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

from ml_engine import run_analysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "../data/customer_support_tickets.csv")

PORT = 8080

analysis_cache = None


def load_analysis():
    global analysis_cache
    analysis_cache, _= run_analysis(DATA_FILE)


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/api/analysis":

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            self.wfile.write(json.dumps(analysis_cache).encode())

        else:

            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        return


if __name__ == "__main__":

    print("Running ML analysis...")

    load_analysis()

    print("Starting API server...")

    server = HTTPServer(("0.0.0.0", PORT), Handler)

    print(f"API running at http://localhost:{PORT}")

    server.serve_forever()