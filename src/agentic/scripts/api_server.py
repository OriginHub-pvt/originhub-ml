#!/usr/bin/env python3
"""
API server runner for OriginHub Agentic System.
Start the FastAPI server to enable UI connections.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

from src.agentic.api.app import run

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print("\n" + "="*60)
    print("OriginHub Agentic System API")
    print("="*60)
    print(f"\nStarting API server on {host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Alternative Docs: http://{host}:{port}/redoc")
    print("\n" + "="*60 + "\n")
    
    run(host=host, port=port)
