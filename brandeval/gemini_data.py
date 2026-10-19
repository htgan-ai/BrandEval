import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

api_key = os.environ.get('BRANDEVAL_API_KEY') or os.environ.get('GEMINI_API_KEY', '')
url = os.environ.get('BRANDEVAL_API_URL') or os.environ.get('GEMINI_API_URL', '')
headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
DEFAULT_TIMEOUT = float(os.environ.get('GEMINI_API_TIMEOUT', '120'))
DISABLE_NETWORK = os.environ.get('BRANDEVAL_API_DISABLE_NETWORK', os.environ.get('GEMINI_API_DISABLE_NETWORK', '1')) != '0'
MAX_RETRIES = int(os.environ.get('GEMINI_API_MAX_RETRIES', '5'))
DEFAULT_MODEL = os.environ.get('GEMINI_API_DEFAULT_MODEL', 'gemini-2.5-flash-lite')
# Force lower concurrency to avoid API limits
MAX_CONCURRENCY = int(os.environ.get('GEMINI_API_MAX_CONCURRENCY', '1'))
# _executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)

# Setup session with retry
_session = requests.Session()
# Disable proxy to avoid system proxy issues
_session.trust_env = False
retries = Retry(
    total=MAX_RETRIES, 
    backoff_factor=2, 
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST", "GET"]
)
_session.mount('https://', HTTPAdapter(max_retries=retries))

prompt = ""

def construct_payload(user_input, history, model_name=None):
    messages = [{"role": 'system', "content": prompt}, {"role": 'user', "content": user_input}]
    for h in history:
        messages.append({"role": 'user', "content": h[0]})
        messages.append({"role": 'assistant', "content": h[1]})
    return {'model': model_name or DEFAULT_MODEL, 'messages': messages}

def _do_request(payload):
    last_error = None
    # Add delay to avoid rate limits
    # time.sleep(2)
    
    
    # Using session with retry adapter for connection errors
    # But we still keep the outer loop for other exceptions or if adapter fails
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _session.post(url, headers=headers, data=json.dumps(payload), timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            # 检查API返回是否包含预期字段
            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Invalid API response structure: {data}")
            content = data['choices'][0]['message']['content']
            return content
        except Exception as exc:
            last_error = exc
            print(f"[GeminiAPI] attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                wait_time = 2 * attempt # Increase wait time
                print(f"[GeminiAPI] Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    error_msg = f'Gemini API failed after {MAX_RETRIES} attempts: {last_error}'
    print(f"[GeminiAPI] {error_msg}")
    raise RuntimeError(error_msg)

def run(user_input, history, model_name=None):
    if DISABLE_NETWORK:
        raise RuntimeError('Network access disabled. Set BRANDEVAL_API_DISABLE_NETWORK=0 to enable it.')
    if not url:
        raise RuntimeError('Set BRANDEVAL_API_URL to an OpenAI-compatible chat-completions endpoint.')
    if not api_key:
        raise RuntimeError('Set BRANDEVAL_API_KEY before enabling network access.')
    
    
    payload = construct_payload(user_input, history, model_name=model_name)
    # future = _executor.submit(_do_request, payload)
    # text = future.result()
    # Run directly to avoid thread pool issues and enforce serial execution
    text = _do_request(payload)
    return history + [(user_input, text)], text

if __name__ == '__main__':
    history = []
    history, response = run('Hello Gemini', history)
    print('Response:', response)
