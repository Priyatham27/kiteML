"""
commands/serve.py — CLI command for starting an inference server.
"""
import argparse
import os

from kiteml.cli.ui.colors import print_step, print_info, print_error
from kiteml.deployment.packaging import load_bundle
from kiteml.output.result import Result


def setup_serve_parser(subparsers):
    parser = subparsers.add_parser("serve", help="Start a REST API server for a trained model")
    parser.add_argument("model", type=str, help="Path to the .kiteml bundle")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.set_defaults(func=run_serve)


def run_serve(args):
    if not os.path.exists(args.model):
        print_error(f"Model bundle not found: {args.model}")
        return 1

    print_info(f"Loading model bundle from {args.model}...")
    try:
        from kiteml.deployment.model_server import serve as serve_model
        # We need a dummy Result or we can pass the bundle path directly if model_server supports it.
        # Actually, our model_server.serve takes a Result object OR we can construct one.
        # Let's import the server logic that can run standalone
        from kiteml.deployment.model_server import serve_bundle
        
        print_step(f"Starting server on {args.host}:{args.port}")
        serve_bundle(args.model, host=args.host, port=args.port)
        
    except ImportError as e:
        print_error(f"Serving requires extra dependencies: {e}")
        print_info("Run `pip install fastapi uvicorn`")
        return 1
    except Exception as e:
        print_error(f"Server failed to start: {e}")
        return 1

    return 0
