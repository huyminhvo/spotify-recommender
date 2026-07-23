from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.spotify_auth import (
    DEFAULT_EVALUATION_TOKEN_CACHE,
    create_cached_user_oauth,
    get_spotify_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authorize Spotify and cache a refreshable token for local evaluation."
    )
    parser.add_argument(
        "--cache-path",
        default=str(DEFAULT_EVALUATION_TOKEN_CACHE),
        help="Local token-cache path (default: .spotify_cache/evaluation-token.json).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the authorization URL without trying to open a browser.",
    )
    return parser.parse_args()


def authorize(cache_path: str | Path, open_browser: bool = True) -> str:
    oauth = create_cached_user_oauth(get_spotify_config(), cache_path)
    authorize_url = oauth.get_authorize_url()

    print("Open this URL and approve access:\n")
    print(authorize_url)
    if open_browser and not webbrowser.open(authorize_url):
        print("\n[warning] Could not open a browser automatically; use the URL above.")

    print(
        "\nAfter approval, Spotify redirects to your configured URI. "
        "Copy the entire URL from the browser address bar."
    )
    redirected_url = input("Redirected URL: ").strip()
    code = oauth.parse_response_code(redirected_url)
    if not code:
        raise ValueError("The redirected URL did not contain a Spotify authorization code.")

    token_info = oauth.get_access_token(code, check_cache=False)
    if not token_info or not token_info.get("access_token"):
        raise RuntimeError("Spotify did not return an access token.")

    user = oauth.validate_token(token_info)
    if not user:
        raise RuntimeError("Spotify returned a token that could not be validated.")
    return str(Path(cache_path).expanduser().resolve())


def main() -> None:
    args = parse_args()
    try:
        cache_path = authorize(args.cache_path, open_browser=not args.no_browser)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Authorization failed: {exc}") from exc
    print(f"\n[authorized] Refreshable Spotify credentials cached at {cache_path}")
    print("You can now run scripts/build_evaluation_dataset.py without a raw access token.")


if __name__ == "__main__":
    main()
