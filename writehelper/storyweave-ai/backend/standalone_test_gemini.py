"""
Standalone Gemini API Connectivity Test

Systematically tests connectivity to the Google Gemini API using google-genai,
verifies environment configuration, prints safe diagnostics, and handles specific error cases.
"""

import os
import sys
import time
from pathlib import Path

# 1. Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    print(f"[OK] Loaded .env from: {env_path}", flush=True)
else:
    print(f"[WARN] No .env found at {env_path}", flush=True)

# 2. Check SDK installation
try:
    import google.genai as genai
    from google.genai import types, errors
    import httpx
    print(f"[OK] google-genai SDK installed. Version: {getattr(genai, '__version__', 'unknown')}", flush=True)
except ImportError as e:
    print(f"[FAIL] google-genai SDK not installed or failed to import: {e}", flush=True)
    sys.exit(1)

# 3. Check environment variables
api_key = os.getenv("GEMINI_API_KEY", "")
model_name = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

if not api_key:
    print("[FAIL] GEMINI_API_KEY is missing or empty in .env", flush=True)
    sys.exit(1)

# Print safe masked key info
masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
print(f"[OK] GEMINI_API_KEY loaded: {masked_key} (length: {len(api_key)})", flush=True)
print(f"[OK] Target model: {model_name}", flush=True)

# 4. Initialize client with explicit timeout
timeout_seconds = 20.0
print(f"[INFO] Initializing Gemini Client with timeout={timeout_seconds}s...", flush=True)
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
)

# 5. Run connectivity test
prompt = "Reply with exactly: GEMINI_OK"
print(f"[INFO] Sending test prompt: '{prompt}' to model '{model_name}'...", flush=True)
start_time = time.time()

try:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    elapsed = time.time() - start_time
    reply = response.text.strip() if response and response.text else "<empty>"
    print(f"[SUCCESS] Gemini responded in {elapsed:.2f}s:", flush=True)
    print(f"--- Response ---")
    print(reply)
    print(f"----------------")
    print(f"[PASS] Real Gemini connectivity verified successfully!", flush=True)

except errors.ClientError as e:
    elapsed = time.time() - start_time
    code = getattr(e, "code", None)
    msg = str(e)
    print(f"[FAIL] ClientError ({code}) after {elapsed:.2f}s:", flush=True)
    if code in (401, 403):
        print(f"  -> Authentication Error: The provided GEMINI_API_KEY is invalid or unauthorized.", flush=True)
    elif code == 404:
        print(f"  -> Model Not Found (404): Model '{model_name}' is not recognized or retired.", flush=True)
        print(f"     Recommended: Check available models with client.models.list().", flush=True)
    elif code == 429:
        print(f"  -> Quota / Rate Limit (429): Quota exceeded for this API key.", flush=True)
    else:
        print(f"  -> Client error details: {msg}", flush=True)

except errors.ServerError as e:
    elapsed = time.time() - start_time
    print(f"[FAIL] ServerError from Gemini after {elapsed:.2f}s: {e}", flush=True)

except (httpx.TimeoutException, TimeoutError) as e:
    elapsed = time.time() - start_time
    print(f"[FAIL] Request Timeout after {elapsed:.2f}s: Request to Gemini timed out.", flush=True)

except Exception as e:
    elapsed = time.time() - start_time
    print(f"[FAIL] Unexpected error ({type(e).__name__}) after {elapsed:.2f}s: {e}", flush=True)
    import traceback
    traceback.print_exc()
