"""
model_server.py — One-command inference server for KiteML models.

Launches a FastAPI server directly from a Result object or a .kiteml bundle.
Requires: pip install fastapi uvicorn
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    reload: bool = False


def serve(
    result: Any,
    host: str = "0.0.0.0",
    port: int = 8000,
    background: bool = False,
    open_browser: bool = False,
) -> None:
    """
    Launch a FastAPI inference server from a KiteML Result.

    Parameters
    ----------
    result : Result
        Fitted KiteML result.
    host : str
        Server host. Default ``'0.0.0.0'``.
    port : int
        Server port. Default ``8000``.
    background : bool
        If True, run in a background thread (for notebooks/testing).
    open_browser : bool
        If True, open Swagger UI in browser after start.

    Examples
    --------
    >>> result.serve(port=8000)
    """
    try:
        import fastapi
        import uvicorn
    except ImportError:
        raise ImportError(
            "Serving requires FastAPI and Uvicorn.\n"
            "Install with: pip install fastapi uvicorn"
        )

    from kiteml.serving.fastapi_app import create_app

    app = create_app(result)
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    print(f"\n🚀 KiteML Model Server")
    print(f"   Model     : {result.model_name}")
    print(f"   Type      : {result.problem_type}")
    print(f"   API URL   : {url}")
    print(f"   Swagger   : {url}/docs")
    print(f"   Health    : {url}/health")
    print(f"   Press Ctrl+C to stop\n")

    if open_browser:
        def _open():
            time.sleep(1.5)
            import webbrowser
            webbrowser.open(f"{url}/docs")
        threading.Thread(target=_open, daemon=True).start()

    if background:
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        time.sleep(0.5)  # allow startup
        return

    server.run()


def serve_bundle(
    bundle_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """
    Launch inference server directly from a .kiteml bundle directory.

    Parameters
    ----------
    bundle_path : str
        Path to a .kiteml bundle directory.
    host : str
    port : int
    """
    from kiteml.deployment.realtime_inference import RealtimeInferenceEngine
    engine = RealtimeInferenceEngine.from_bundle(bundle_path)

    try:
        import uvicorn
        from kiteml.serving.fastapi_app import create_app_from_engine
    except ImportError:
        raise ImportError("Install fastapi uvicorn for serving.")

    app = create_app_from_engine(engine)
    print(f"🚀 Serving bundle from {bundle_path} at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
