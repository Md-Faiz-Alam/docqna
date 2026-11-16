"""Smoke-test a running DocQ&A backend before it receives real traffic.

The deploy script (infra/azure/deploy.sh) runs the full test against the
staging app, and only if every check passes deploys the same image to
production, which it then checks with --quick.

    python scripts/smoke_test.py https://<staging-address> --origin https://docqna-lake.vercel.app --expect-build <sha>
    python scripts/smoke_test.py https://<production-address> --quick --expect-build <sha>

Checks, in order:
1. /health answers (waits for a cold start, up to --wait seconds); with
   --expect-build, also waits until /info reports that commit, so the
   checks never run against the previous version
2. /ready: vector store, registry and LLM are reachable
3. CORS: the website's origin may call the API (with --origin)
4. a question in an empty session is refused without calling the LLM
5. sample documents load, and a real question gets a cited, signed answer
   (one LLM call; skipped with --quick)
6. the per-IP rate limit holds when every request uses a new session ID and
   a faked X-Forwarded-For header (skipped with --quick)

Standard library only, so it runs on a CI runner or in Azure Cloud Shell
without installing anything. Exits non-zero on the first failed check.
"""

import argparse
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from typing import Any

SESSION_HEADER = "X-DocQNA-Session-ID"
SAMPLE_QUESTION = "How many hours is the standard working week?"
# Settings defaults: 20 questions per minute per session, x3 per IP.
IP_QUESTION_LIMIT = 60


class SmokeTestFailure(Exception):
    pass


def call(
    method: str,
    url: str,
    *,
    body: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = 90.0,
) -> tuple[int, dict[str, str], Any]:
    """(status, lower-cased headers, parsed JSON or raw text). Never raises on HTTP errors."""

    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("User-Agent", "docqna-smoke-test")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw, response_headers = response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        status, raw, response_headers = error.code, error.read(), error.headers
    text = raw.decode("utf-8", errors="replace")
    try:
        payload: Any = json.loads(text) if text else None
    except json.JSONDecodeError:
        payload = text
    return status, {k.lower(): v for k, v in response_headers.items()}, payload


def new_session() -> str:
    return secrets.token_hex(20)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def wait_for_health(base: str, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, _, payload = call("GET", f"{base}/health", timeout=30)
            if status == 200:
                print(f"ok    /health ({payload})")
                return
            last = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last = str(error)
        time.sleep(5)
    raise SmokeTestFailure(f"/health did not answer within {wait_seconds:.0f}s (last: {last})")


def wait_for_build(base: str, expected: str, wait_seconds: float) -> None:
    """Wait until /info reports the expected commit (the new revision is serving)."""

    deadline = time.monotonic() + wait_seconds
    seen = None
    while time.monotonic() < deadline:
        try:
            status, _, payload = call("GET", f"{base}/info", timeout=30)
            if status == 200 and isinstance(payload, dict):
                seen = payload.get("build")
                if seen == expected:
                    print(f"ok    /info reports build {expected}")
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(5)
    raise SmokeTestFailure(
        f"/info still reports build {seen!r} after {wait_seconds:.0f}s, expected {expected!r}"
    )


def check_ready(base: str) -> None:
    status, _, payload = call("GET", f"{base}/ready")
    check(status == 200, f"/ready returned HTTP {status}: {payload}")
    print(f"ok    /ready {payload.get('checks') if isinstance(payload, dict) else payload}")


def check_cors(base: str, origin: str) -> None:
    status, headers, _ = call(
        "OPTIONS",
        f"{base}/qa",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{SESSION_HEADER.lower()}",
        },
    )
    allowed = headers.get("access-control-allow-origin")
    check(
        status == 200 and allowed == origin,
        f"CORS preflight from {origin}: HTTP {status}, allow-origin={allowed!r}",
    )
    print(f"ok    CORS allows {origin}")


def check_empty_session_refuses(base: str) -> None:
    status, _, payload = call(
        "POST",
        f"{base}/qa",
        body={"question": "smoke test"},
        headers={SESSION_HEADER: new_session()},
    )
    check(status == 200, f"/qa in an empty session returned HTTP {status}: {payload}")
    check(payload.get("abstained") is True, f"/qa in an empty session did not refuse: {payload}")
    print("ok    empty session is refused")


def check_real_answer(base: str) -> None:
    session = {SESSION_HEADER: new_session()}
    status, _, payload = call("POST", f"{base}/documents/samples", headers=session, timeout=180)
    check(status == 200, f"loading sample documents returned HTTP {status}: {payload}")
    check(payload.get("count", 0) > 0, f"no sample documents were loaded: {payload}")
    print(f"ok    {payload['count']} sample documents loaded")

    # The hosted LLM's free tier can briefly rate-limit (503); retry before failing.
    for attempt in range(1, 4):
        status, _, answer = call(
            "POST", f"{base}/qa", body={"question": SAMPLE_QUESTION}, headers=session, timeout=120
        )
        if status != 503:
            break
        print(f"...   LLM busy (attempt {attempt}), retrying in 15s")
        time.sleep(15)
    check(status == 200, f"/qa returned HTTP {status}: {answer}")
    check(answer.get("abstained") is False, f"the sample question was refused: {answer}")
    check(bool(answer.get("sources")), f"the answer has no sources: {answer}")
    check(bool(answer.get("answer_signature")), f"the answer is not signed: {answer}")
    print(f"ok    cited answer: {answer['answer'][:100]!r}")


def check_ip_rate_limit(base: str) -> None:
    """Every request uses a new session and a fake X-Forwarded-For; the IP limit must still apply."""

    for number in range(1, IP_QUESTION_LIMIT + 20):
        status, _, payload = call(
            "POST",
            f"{base}/qa",
            body={"question": "rate limit check"},
            headers={SESSION_HEADER: new_session(), "X-Forwarded-For": f"10.{number}.0.1"},
            timeout=60,
        )
        if status == 429:
            print(
                f"ok    per-IP rate limit applied at request {number} despite new sessions and faked IPs"
            )
            return
        check(
            status == 200, f"rate-limit check: request {number} returned HTTP {status}: {payload}"
        )
    raise SmokeTestFailure(
        f"no 429 after {IP_QUESTION_LIMIT + 19} requests: the per-IP limit can be dodged "
        "(check TRUSTED_PROXY_HOPS)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "base_url", help="for example https://docqna-api-staging.<env>.azurecontainerapps.io"
    )
    parser.add_argument("--origin", help="website origin whose CORS access is checked")
    parser.add_argument(
        "--wait", type=float, default=300.0, help="seconds to wait for a cold start"
    )
    parser.add_argument(
        "--quick", action="store_true", help="skip the LLM answer and rate-limit checks"
    )
    parser.add_argument(
        "--expect-build", help="commit SHA that /info must report before the checks run"
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Smoke test: {base}")
    try:
        wait_for_health(base, args.wait)
        if args.expect_build:
            wait_for_build(base, args.expect_build, args.wait)
        check_ready(base)
        if args.origin:
            check_cors(base, args.origin)
        check_empty_session_refuses(base)
        if not args.quick:
            check_real_answer(base)
            check_ip_rate_limit(base)
    except SmokeTestFailure as failure:
        print(f"FAIL  {failure}")
        return 1
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
