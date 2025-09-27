import os, requests, json, time

JUDGE0_URL = "https://judge0-ce.p.rapidapi.com/submissions"
API_KEY = os.getenv("JUDGE0_RAPIDAPI_KEY", "").strip()
HEADERS = {
    "Content-Type": "application/json",
    "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
    "X-RapidAPI-Key": "91dd6d88ecmsh26fe0cc2ceb8db1p183040jsn71c8c1dc441a"
}

def normalize_output(s):
    if s is None:
        return ""
    # remove trailing newlines/spaces; normalize line endings; strip each line
    lines = [line.rstrip() for line in s.replace("\r\n", "\n").split("\n")]
    # remove any empty trailing lines
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)

def float_equal(a_str, b_str, tol=1e-6):
    try:
        return abs(float(a_str) - float(b_str)) <= tol
    except Exception:
        return False

def compare_outputs(actual, expected, numeric_tolerance=False, tol=1e-6):
    a = normalize_output(actual)
    e = normalize_output(expected)
    if numeric_tolerance:
        # linewise numeric tolerant compare if both look numeric
        a_lines = a.splitlines()
        e_lines = e.splitlines()
        if len(a_lines) != len(e_lines):
            return False
        for al, el in zip(a_lines, e_lines):
            if al == el:
                continue
            if not float_equal(al, el, tol):
                return False
        return True
    else:
        return a == e

def run_on_judge0(source_code, stdin, language_id=71, base64_encoded=False, wait=True):
    params = {
        "base64_encoded": "false",
        "wait": "true" if wait else "false"
    }
    payload = {
        "source_code": source_code,
        "language_id": language_id,
        "stdin": stdin
    }
    resp = requests.post(JUDGE0_URL, params=params, json=payload, headers=HEADERS, timeout=30)
    # caller should check resp.status_code (403 etc)
    resp.raise_for_status()
    return resp.json()

def judge_submission(source_code, tests, numeric_tolerance=False, tol=1e-6):
    """
    source_code: string containing user's program
    tests: list of dicts: [{"input": "...", "expected": "..."}, ...]
    returns: list of results per test + summary
    """
    results = []
    passed = 0
    for i, t in enumerate(tests, start=1):
        try:
            res = run_on_judge0(source_code, t["input"])
        except requests.HTTPError as e:
            results.append({"test": i, "status": "error", "detail": str(e), "body": getattr(e.response, "text", "")})
            continue
        # If wait=true, res is final result
        stdout = res.get("stdout") or ""
        stderr = res.get("stderr") or ""
        status_desc = res.get("status", {}).get("description", "")
        status_id = res.get("status", {}).get("id", 0)

        # If runtime error or compile error, mark fail and include messages
        if status_id != 3:  # 3 usually means "Accepted" in Judge0
            results.append({
                "test": i,
                "status": "failed",
                "reason": status_desc,
                "stdout": stdout,
                "stderr": stderr,
                "token": res.get("token")
            })
            continue

        ok = compare_outputs(stdout, t["expected"], numeric_tolerance, tol)
        results.append({
            "test": i,
            "status": "passed" if ok else "failed",
            "stdout": stdout,
            "expected": t["expected"]
        })
        if ok:
            passed += 1
        # small delay to avoid hitting quota limits quickly
        time.sleep(0.2)

    summary = {"total": len(tests), "passed": passed}
    return results, summary

# ----- Example usage -----
if __name__ == "__main__":
    # load user's source (or read file)
    with open("my_code.py", "r", encoding="utf-8") as f:
        user_src = f.read()

    tests = [
        {"input": "5\n1 5 2 9 7\n", "expected": "Built-in sort: [1, 2, 5, 7, 9]\nBubble sort: [1, 2, 5, 7, 9]"},
        {"input": "5\n1 5 2 9 7\n", "expected": "Built-in sort: [1, 2, 5, 7, 9]\nBubble sort: [1, 2, 5, 8, 9]"},
        # add more test cases...
    ]

    res, summary = judge_submission(user_src, tests)
    print(json.dumps({"summary": summary, "results": res}, indent=2))
