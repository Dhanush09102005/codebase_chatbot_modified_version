#!/usr/bin/env python3
"""
Main entry point for the Code Q&A Assistant API.

Run with: uvicorn main:app --reload
Or: python main.py
"""
import uvicorn
from app.api.app import app
from app.core import settings


if __name__ == "__main__":
    uvicorn.run(
        "app.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
