# test_auth.py
from google.cloud import bigquery
from dotenv import load_dotenv
import os

load_dotenv() # Load the .env file

print(f"1. Looking for key at: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

try:
    # This tries to connect to Google using that key
    client = bigquery.Client()
    print(f"2. SUCCESS! Connected to project: {client.project}")
except Exception as e:
    print(f"2. FAILED: {e}")