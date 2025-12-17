"""
main.py — Application entry point for Rubik's Cube Solver webview UI
=================================================================

This module launches the pywebview-based GUI and wires the Python <-> JS API.

Features & behavior:
 - Creates a single webview window hosting `templates/index.html`.
 - Supports CLI flags for debug mode, fullscreen, resizable, and window size.
 - Runs kociemba and camera logic in the API object (API is responsible for
   exposing safe methods to the frontend).
 - Provides robust startup/shutdown: ensures API.shutdown() is called on exit,
   handles SIGINT/SIGTERM, and logs exceptions.
 - Debug mode is explicitly opt-in (`--debug`)

------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. MIT License.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

import webview

from api import API

DEFAULT_HTML = "templates/index.html"

# if --debug is used.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.
    """
    p = argparse.ArgumentParser(
        description="Rubik's Cube Solver",
        allow_abbrev=False,
    )

    p.add_argument("--debug", action="store_true", help="Enable webview debug mode.")
    p.add_argument("--fullscreen", action="store_true", help="Start window in fullscreen.")
    p.set_defaults(fullscreen=True)

    # Explicit flags for enabling/disabling resizable mode
    p.add_argument("--resizable", dest="resizable", action="store_true", help="Allow window resizing.")
    p.add_argument("--no-resize", dest="resizable", action="store_false", help="Disable window resizing.")
    p.set_defaults(resizable=True)

    # HTML entrypoint
    p.add_argument(
        "--html",
        default=DEFAULT_HTML,
        help="HTML file to load in the webview."
    )

    return p


def _install_signal_handlers(shutdown_callable):
    """
    Install safe signal handlers for SIGINT & SIGTERM.

    When triggered, the handler:
      - Logs the event
      - Calls the provided shutdown function
      - Exits cleanly using SystemExit
    """
    def _handler(signum, frame):
        logger.info("Received signal %s, initiating shutdown...", signum)
        try:
            shutdown_callable()
        except Exception as e:
            logger.exception("Error during shutdown handler: %s", e)
        raise SystemExit(0)

    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig_name):
            signal.signal(getattr(signal, sig_name), _handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point.

    Supports direct CLI invocation or programmatic use via:
        main(["--fullscreen"])
    
    Returns integer exit code.
    """
    args = create_arg_parser().parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled (verbose logging + webview debugging).")

    # Resolve HTML path relative to this file if not absolute
    html_path = Path(args.html)
    if not html_path.is_absolute():
        html_path = (Path(__file__).parent / html_path).resolve()

    if not html_path.exists():
        logger.error("HTML file not found: %s", html_path)
        return 2

    api: Optional[API] = None
    try:
        api = API()
    except Exception as e:
        logger.exception("Failed to initialize API: %s", e)
        return 3

    # Shutdown wrapper
    def _shutdown_safely():
        nonlocal api
        if api:
            try:
                api.shutdown()
            except Exception as e:
                logger.exception("Exception during API.shutdown(): %s", e)
            finally:
                api = None

    # Register shutdown handlers
    atexit.register(_shutdown_safely)
    _install_signal_handlers(_shutdown_safely)

    window_title = "Rubik's Cube Solver"
    window_url = str(html_path)

    try:
        create_kwargs = {
            "title": window_title,
            "url": window_url,
            "js_api": api,
            "resizable": bool(args.resizable),
            "fullscreen": bool(args.fullscreen),
        }

        logger.info(
            "Creating window: %s (size=%s) fullscreen=%s resizable=%s",
            window_url,
            f"{create_kwargs.get('width', 'auto')}x{create_kwargs.get('height', 'auto')}",
            create_kwargs["fullscreen"],
            create_kwargs["resizable"],
        )

        webview.create_window(**create_kwargs)

        logger.info("Starting webview (debug=%s)...", args.debug)
        webview.start(debug=args.debug)

    except SystemExit:
        # Triggered by signal handlers — normal exit path
        logger.info("Shutdown requested (SystemExit).")
    except Exception as e:
        logger.exception("Unhandled exception while running webview: %s", e)
        try:
            _shutdown_safely()
        except Exception:
            pass
        return 1
    finally:
        # Final shutdown attempt
        try:
            _shutdown_safely()
        except Exception:
            logger.exception("Error during final shutdown.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
