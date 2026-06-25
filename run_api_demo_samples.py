import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URL = "http://127.0.0.1:5000/predict"
DEFAULT_SAMPLES_FILE = "demo_api_samples.json"


def post_json(url, payload, timeout=10):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.getcode(), json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": body}
        return exc.code, parsed
    except urllib.error.URLError as exc:
        return None, {"error": f"Connection error: {exc.reason}"}


def format_row(values, widths):
    cells = []
    for value, width in zip(values, widths):
        text = str(value)
        if len(text) > width:
            text = text[: width - 3] + "..."
        cells.append(text.ljust(width))
    return " | ".join(cells)


def print_table(rows):
    headers = ["sample", "status", "prediction", "fraud_probability", "error"]

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    separator = "-+-".join("-" * w for w in widths)
    print(format_row(headers, widths))
    print(separator)
    for row in rows:
        print(format_row(row, widths))


def main():
    parser = argparse.ArgumentParser(
        description="Send all samples from demo_api_samples.json to the fraud API and print a summary table."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Prediction API URL")
    parser.add_argument(
        "--samples",
        default=DEFAULT_SAMPLES_FILE,
        help="Path to JSON file containing named sample payloads",
    )
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds")
    args = parser.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"Samples file not found: {samples_path}")
        sys.exit(1)

    with samples_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)

    rows = []
    for sample_name, payload in samples.items():
        status_code, response_body = post_json(args.url, payload, timeout=args.timeout)

        prediction = response_body.get("prediction", "-")
        fraud_probability = response_body.get("fraud_probability", "-")
        if isinstance(fraud_probability, float):
            fraud_probability = f"{fraud_probability:.6f}"
        error = response_body.get("error", "")

        rows.append(
            [
                sample_name,
                status_code if status_code is not None else "-",
                prediction,
                fraud_probability,
                error,
            ]
        )

    print_table(rows)


if __name__ == "__main__":
    main()
