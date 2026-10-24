#!/usr/bin/env python3
"""Container runtime smoke test runner.

Phase 10.1 release validation:
- Validates a live running container instance of the inference API.
- Tests /health liveness and /ready readiness probes with bounded retries.
- Verifies Docker HEALTHCHECK status reaches 'healthy'.
- Verifies application process runs as an unprivileged non-root user.
- Tests POST /v1/predict for standard and cold-start applicant profiles.
- Validates probability bounds, PDO score, risk band, and additive explanations.
- Verifies non-underwriting governance contracts (no approval/rejection decisions).
- Verifies input validation contracts (HTTP 422 on forbidden extra fields).
- Ensures no applicant input payloads or sensitive values are printed in logs.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _make_http_request(
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    """Execute HTTP request using standard library urllib without logging payloads."""
    body_bytes = None
    headers = {"Accept": "application/json"}
    if data is not None:
        body_bytes = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            content = resp.read().decode("utf-8")
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = content
            return status_code, parsed
    except urllib.error.HTTPError as err:
        status_code = err.code
        content = err.read().decode("utf-8")
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = content
        return status_code, parsed


def check_liveness(base_url: str, max_wait: float, interval: float) -> None:
    """Verify GET /health returns HTTP 200 within bounded retries."""
    url = f"{base_url}/health"
    deadline = time.time() + max_wait
    last_err = ""
    while time.time() < deadline:
        try:
            status_code, body = _make_http_request(url, method="GET")
            if status_code == 200 and isinstance(body, dict) and body.get("status") == "healthy":
                print("[PASS] GET /health returned HTTP 200 {'status': 'healthy'}")
                return
            last_err = f"Unexpected status {status_code} or body: {body}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(interval)
    raise RuntimeError(f"Liveness check failed after {max_wait}s timeout: {last_err}")


def check_readiness(base_url: str, max_wait: float, interval: float) -> None:
    """Verify GET /ready returns HTTP 200 with model_loaded: true within bounded retries."""
    url = f"{base_url}/ready"
    deadline = time.time() + max_wait
    last_err = ""
    while time.time() < deadline:
        try:
            status_code, body = _make_http_request(url, method="GET")
            if status_code == 200 and isinstance(body, dict):
                if body.get("status") == "ready" and body.get("model_loaded") is True:
                    print("[PASS] GET /ready returned HTTP 200 with status='ready', model_loaded=True")
                    return
            last_err = f"Status {status_code}, body: {body}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(interval)
    raise RuntimeError(f"Readiness check failed after {max_wait}s timeout: {last_err}")


def check_docker_health_and_user(container_name: str, max_wait: float, interval: float) -> None:
    """Verify Docker container health reaches 'healthy' and process is non-root."""
    print(f"[*] Inspecting Docker container '{container_name}'...")

    # 1. Verify healthcheck status
    deadline = time.time() + max_wait
    last_status = "unknown"
    while time.time() < deadline:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            health_status = proc.stdout.strip().lower()
            last_status = health_status
            if health_status == "healthy":
                print(f"[PASS] Docker container '{container_name}' health status is 'healthy'")
                break
            if health_status == "unhealthy":
                raise RuntimeError(f"Docker container '{container_name}' reported health status 'unhealthy'")
        else:
            last_status = f"inspect failed: {proc.stderr.strip()}"
        time.sleep(interval)
    else:
        raise RuntimeError(f"Docker health check did not reach 'healthy' within {max_wait}s (status: {last_status})")

    # 2. Verify non-root user execution
    proc_user = subprocess.run(
        ["docker", "exec", container_name, "whoami"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc_user.returncode != 0:
        raise RuntimeError(f"Failed to query runtime username: {proc_user.stderr.strip()}")
    username = proc_user.stdout.strip()

    proc_uid = subprocess.run(
        ["docker", "exec", container_name, "id", "-u"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc_uid.returncode != 0:
        raise RuntimeError(f"Failed to query runtime UID: {proc_uid.stderr.strip()}")
    uid_str = proc_uid.stdout.strip()

    if username == "root" or uid_str == "0":
        raise RuntimeError(f"Container process is running as root (user: {username}, UID: {uid_str})!")
    print(f"[PASS] Container runtime process is non-root (user: '{username}', UID: {uid_str})")


def test_standard_prediction(base_url: str) -> None:
    """Verify POST /v1/predict with standard applicant returns HTTP 200 and valid schema."""
    url = f"{base_url}/v1/predict"
    payload = {
        "age_years": 32,
        "monthly_income_inr": 55000.0,
        "employment_tenure_years": 4.5,
        "debt_to_income_ratio": 0.28,
        "existing_loan_count": 1,
        "credit_utilization_ratio": 0.35,
        "prior_delinquencies_24m": 0,
        "bureau_history_available": 1,
        "bureau_score": 720.0,
        "utility_payment_timeliness": 0.95,
        "rental_payment_timeliness": 0.92,
        "digital_payment_regularity": 0.88,
        "cashflow_stability_index": 0.75,
        "alternative_history_months": 18,
    }

    status_code, body = _make_http_request(url, method="POST", data=payload)
    if status_code != 200:
        raise RuntimeError(f"Standard prediction returned HTTP {status_code}: {body}")

    if not isinstance(body, dict):
        raise TypeError(f"Expected dict response body, got {type(body).__name__}")

    # Validate required fields
    required_keys = [
        "model_identifier",
        "schema_version",
        "default_probability_12m",
        "risk_band",
        "scoring",
        "top_risk_increasing_factors",
        "top_risk_decreasing_factors",
        "disclaimer",
        "educational_notice",
    ]
    for key in required_keys:
        if key not in body:
            raise KeyError(f"Missing required response field: '{key}'")

    prob = body["default_probability_12m"]
    if not isinstance(prob, (int, float)) or not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
        raise ValueError(f"Invalid probability value: {prob}")

    scoring = body["scoring"]
    if not isinstance(scoring, dict) or "display_score" not in scoring:
        raise ValueError(f"Invalid scoring structure: {scoring}")

    display_score = scoring["display_score"]
    if not isinstance(display_score, int) or not (300 <= display_score <= 850):
        raise ValueError(f"Invalid display score: {display_score}")

    valid_bands = {
        "Lower model-estimated risk",
        "Middle model-estimated risk",
        "Higher model-estimated risk",
    }
    risk_band = body["risk_band"]
    if risk_band not in valid_bands:
        raise ValueError(f"Unexpected risk band: '{risk_band}'")

    increasing = body["top_risk_increasing_factors"]
    decreasing = body["top_risk_decreasing_factors"]
    if not isinstance(increasing, list) or len(increasing) > 3:
        raise ValueError(f"Invalid top_risk_increasing_factors length: {len(increasing)}")
    if not isinstance(decreasing, list) or len(decreasing) > 3:
        raise ValueError(f"Invalid top_risk_decreasing_factors length: {len(decreasing)}")

    for factor in increasing + decreasing:
        if not isinstance(factor, dict) or "feature" not in factor or "contribution" not in factor:
            raise ValueError(f"Malformed factor item: {factor}")
        if not isinstance(factor["contribution"], (int, float)) or not math.isfinite(factor["contribution"]):
            raise ValueError(f"Non-finite factor contribution: {factor}")

    # Verify disclaimers present
    disclaimer = body["disclaimer"]
    educational_notice = body["educational_notice"]
    if "synthetic data" not in disclaimer or "not validated for real borrowers" not in disclaimer:
        raise ValueError(f"Disclaimer text does not contain required synthetic notice: '{disclaimer}'")
    if "do not constitute an approval" not in educational_notice:
        raise ValueError(f"Educational notice does not contain non-underwriting clause: '{educational_notice}'")

    # Verify no underwriting authority claim
    forbidden_keys = [
        "approved",
        "declined",
        "approval",
        "decision",
        "credit_decision",
        "pricing",
        "interest_rate",
        "credit_limit",
        "underwriting_result",
    ]
    for key in forbidden_keys:
        if key in body:
            raise ValueError(f"Response contains forbidden underwriting decision field: '{key}'")

    print(
        f"[PASS] POST /v1/predict (standard applicant): HTTP 200, "
        f"default_probability_12m={prob:.4f}, display_score={display_score}, risk_band='{risk_band}'"
    )
    print("[PASS] Governance verification: response contains no underwriting decision fields")


def test_cold_start_prediction(base_url: str) -> None:
    """Verify POST /v1/predict accepts thin-file applicant with no bureau and null alternative data."""
    url = f"{base_url}/v1/predict"
    payload = {
        "age_years": 24,
        "monthly_income_inr": 25000.0,
        "employment_tenure_years": 1.0,
        "debt_to_income_ratio": 0.15,
        "existing_loan_count": 0,
        "credit_utilization_ratio": 0.10,
        "prior_delinquencies_24m": 0,
        "bureau_history_available": 0,
        "bureau_score": None,
        "utility_payment_timeliness": None,
        "rental_payment_timeliness": None,
        "digital_payment_regularity": None,
        "cashflow_stability_index": None,
        "alternative_history_months": None,
    }

    status_code, body = _make_http_request(url, method="POST", data=payload)
    if status_code != 200 or not isinstance(body, dict):
        raise RuntimeError(f"Cold-start prediction returned HTTP {status_code}: {body}")

    prob = body.get("default_probability_12m")
    if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
        raise ValueError(f"Invalid cold-start probability: {prob}")

    score = body.get("scoring", {}).get("display_score")
    if not isinstance(score, int) or not (300 <= score <= 850):
        raise ValueError(f"Invalid cold-start score: {score}")

    risk_band = body.get("risk_band")
    print(
        f"[PASS] POST /v1/predict (cold-start applicant): HTTP 200, "
        f"default_probability_12m={prob:.4f}, display_score={score}, risk_band='{risk_band}'"
    )


def test_input_contract_rejection(base_url: str) -> None:
    """Verify POST /v1/predict strictly rejects extra forbidden fields with sanitized HTTP 422."""
    url = f"{base_url}/v1/predict"
    payload = {
        "applicant_id": "REJECT_ID_999",
        "age_years": 32,
        "monthly_income_inr": 55000.0,
        "employment_tenure_years": 4.5,
        "debt_to_income_ratio": 0.28,
        "existing_loan_count": 1,
        "credit_utilization_ratio": 0.35,
        "prior_delinquencies_24m": 0,
        "bureau_history_available": 1,
        "bureau_score": 720.0,
        "utility_payment_timeliness": 0.95,
        "rental_payment_timeliness": 0.92,
        "digital_payment_regularity": 0.88,
        "cashflow_stability_index": 0.75,
        "alternative_history_months": 18,
    }

    status_code, body = _make_http_request(url, method="POST", data=payload)
    if status_code != 422:
        raise RuntimeError(f"Expected HTTP 422 for extra forbidden field, got {status_code}: {body}")

    # Verify input value is NOT reflected in error response
    resp_str = json.dumps(body) if isinstance(body, (dict, list)) else str(body)
    if "REJECT_ID_999" in resp_str:
        raise ValueError("Error response reflected raw input value; data leakage vulnerability detected!")

    print("[PASS] Contract validation: extra forbidden field correctly rejected with sanitized HTTP 422")


def test_metrics_scrape(base_url: str) -> None:
    """Verify GET /metrics responds with valid Prometheus exposition reflecting prior smoke requests."""
    url = f"{base_url}/metrics"
    status_code, body = _make_http_request(url, method="GET")

    if status_code != 200:
        raise RuntimeError(f"Scraping /metrics failed with HTTP {status_code}: {body}")

    body_text = str(body) if not isinstance(body, str) else body

    # 1. Verify expected metric families are present
    expected_families = [
        "http_requests_total",
        "http_request_duration_seconds",
        "prediction_outcomes_total",
        "model_ready",
        "model_info",
    ]
    for family in expected_families:
        if f"# HELP {family}" not in body_text or f"# TYPE {family}" not in body_text:
            raise ValueError(f"Metrics exposition missing definition for family: '{family}'")

    # 2. Verify request counts reflect the completed smoke requests
    # Smoke run executed /health, /ready, standard /v1/predict (200), cold-start /v1/predict (200), extra field (422)
    if 'http_requests_total{method="GET",route="/health",status="200"}' not in body_text:
        raise ValueError("Metrics missing expected counter for GET /health")
    if 'http_requests_total{method="GET",route="/ready",status="200"}' not in body_text:
        raise ValueError("Metrics missing expected counter for GET /ready")
    if 'http_requests_total{method="POST",route="/v1/predict",status="200"}' not in body_text:
        raise ValueError("Metrics missing expected counter for POST /v1/predict status 200")
    if 'http_requests_total{method="POST",route="/v1/predict",status="422"}' not in body_text:
        raise ValueError("Metrics missing expected counter for POST /v1/predict status 422")

    # 3. Verify prediction outcome counters reflect the smoke run
    if 'prediction_outcomes_total{model="hybrid_logistic_regression",outcome="success"}' not in body_text:
        raise ValueError("Metrics missing expected prediction_outcomes_total outcome=success")
    if 'prediction_outcomes_total{model="hybrid_logistic_regression",outcome="validation_error"}' not in body_text:
        raise ValueError("Metrics missing expected prediction_outcomes_total outcome=validation_error")

    # 4. Verify gauges reflect readiness and model info
    if 'model_ready{model="hybrid_logistic_regression"} 1.0' not in body_text:
        raise ValueError("Metrics missing expected model_ready gauge value 1.0")
    if 'model_info{model="hybrid_logistic_regression",schema_version="v1"} 1.0' not in body_text:
        raise ValueError("Metrics missing expected model_info gauge value 1.0")

    # 5. Verify /metrics scraping is excluded from request metrics
    if 'route="/metrics"' in body_text:
        raise ValueError("Metrics contains self-observing route='/metrics' entry!")

    # 6. Verify zero sensitive test identifier or raw applicant data leakage
    forbidden_values = [
        "REJECT_ID_999",
        "55000.0",
        "720.0",
        "applicant_id",
        "digital_payment_regularity",
        "cashflow_stability_index",
    ]
    for sensitive in forbidden_values:
        if sensitive in body_text:
            raise ValueError(f"Metrics exposition leaked sensitive value or feature name: '{sensitive}'")

    print("[PASS] GET /metrics returned HTTP 200 with all metric families, valid counts, and zero data leakage")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference Container Runtime Smoke Test")
    parser.add_argument("--host", default="127.0.0.1", help="Target host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18765, help="Target port (default: 18765)")
    parser.add_argument("--container-name", default=None, help="Optional container name to inspect")
    parser.add_argument("--max-wait", type=float, default=30.0, help="Max wait seconds for probes (default: 30)")
    parser.add_argument("--interval", type=float, default=1.0, help="Retry interval in seconds (default: 1)")

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print("================================================================================")
    print(f"RUNNING CONTAINER RUNTIME SMOKE TESTS AGAINST {base_url}")
    print("================================================================================")

    try:
        # 1. Liveness check
        check_liveness(base_url, max_wait=args.max_wait, interval=args.interval)

        # 2. Readiness check
        check_readiness(base_url, max_wait=args.max_wait, interval=args.interval)

        # 3. Docker healthcheck and non-root execution check (if container name provided)
        if args.container_name:
            check_docker_health_and_user(args.container_name, max_wait=args.max_wait, interval=args.interval)

        # 4. Standard applicant prediction test
        test_standard_prediction(base_url)

        # 5. Cold-start applicant prediction test
        test_cold_start_prediction(base_url)

        # 6. Extra field contract rejection test
        test_input_contract_rejection(base_url)

        # 7. Privacy-safe Prometheus metrics endpoint scrape test
        test_metrics_scrape(base_url)

        print("================================================================================")
        print("ALL CONTAINER RUNTIME SMOKE TESTS PASSED SUCCESSFULLY")
        print("================================================================================")
        return 0

    except Exception as exc:
        print(f"[FAIL] Smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
