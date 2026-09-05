"""Vercel / local ASGI entrypoint.

Re-exports the FastAPI app so hosts that look for ``main:app`` (Vercel,
uvicorn ``main:app``) find Sentinel without changing package layout.
"""

from sentinel.api.app import app

__all__ = ["app"]
