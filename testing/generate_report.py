"""Runs the test suite and builds the test-case workbook and HTML report from the real results.

    python testing/generate_report.py          # full suite (live Render test skipped)
    python testing/generate_report.py --live   # also runs the cold-start test against Render

Outputs (in testing/):
    results/junit.xml     raw pytest results
    test_cases.xlsx       test cases with Actual and Pass/Fail filled in, plus defects and corrected tests
    test_report.html      the same, as a self-contained report with every test's outcome

Actual and Pass/Fail come only from this run: a case passes when every linked pytest test
passed, fails if any failed, and is "Not run" if they were skipped.
"""
import html
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parent.parent
TESTING = ROOT / "testing"
RESULTS = TESTING / "results"
JUNIT = RESULTS / "junit.xml"
COLUMNS = ["ID", "Requirement", "Type", "Description", "Steps", "Input", "Expected", "Actual",
           "Pass/Fail", "Linked pytest test"]


def run_pytest(live):
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RUN_LIVE_TESTS="1" if live else "0")
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", f"--junitxml={JUNIT}"]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    return completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "no output"


def load_results():
    """{'tests/test_x.py::test_name': [(outcome, seconds, message), ...]} grouped over parameters."""
    results = {}
    for case in ET.parse(JUNIT).getroot().iter("testcase"):
        node = case.get("classname").replace(".", "/") + ".py::" + case.get("name").split("[")[0]
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            child = case.find(tag)
            if child is not None:
                outcome = "failed" if tag in ("failure", "error") else "skipped"
                message = (child.get("message") or child.text or "").strip()
        results.setdefault(node, []).append((outcome, float(case.get("time") or 0), message))
    return results


def measurement_text(kind):
    path = RESULTS / f"{kind}.json"
    if not path.exists():
        return ""
    m = json.loads(path.read_text(encoding="utf-8"))
    if kind == "performance":
        lat = m["latency_ms"]
        return (f" Measured: {m['requests']} requests, server memory {m['rss_after_startup_mb']} MB after startup, "
                f"peak {m['rss_peak_mb']} MB (limit {m['limits']['memory_mb']}); latency median {lat['median_all']} ms, "
                f"p95 {lat['p95_all']} ms, max {lat['max_all']} ms (limit p95 {m['limits']['p95_ms']} ms); "
                f"startup {m['startup_seconds']} s.")
    text = (f" Latest run ({m['checked_at_utc']}): first response {m['first_response_seconds']} s "
            f"({'service was asleep' if m['looked_cold'] else 'service was already awake'}); warm /predict "
            f"{', '.join(str(s) for s in m['warm_predict_seconds'])} s; /explain {m['explain_seconds']} s.")
    history = RESULTS / "live_history.jsonl"
    if history.exists():
        runs = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
        text += " All runs: " + "; ".join(
            f"{r['checked_at_utc']} first {r['first_response_seconds']} s ({'asleep' if r['looked_cold'] else 'awake'})"
            for r in runs) + "."
    return text


def evaluate(case, results):
    runs = [(test, r) for test in case["tests"] for r in results.get(test, [])]
    missing = [t for t in case["tests"] if t not in results]
    if missing:
        return "Not run", f"Linked test not found in this run: {', '.join(missing)}"
    outcomes = Counter(r[0] for _, r in runs)
    if outcomes["failed"]:
        failure = next(f"{t}: {r[2][:300]}" for t, r in runs if r[0] == "failed")
        return "Fail", f"{outcomes['failed']} of {len(runs)} test runs failed. {failure}"
    if outcomes["skipped"] == len(runs):
        return "Not run", f"Skipped: {runs[0][1][2][:200]}"
    actual = f"{outcomes['passed']} of {len(runs)} test runs passed; all assertions in the expected result held."
    if case.get("measurement"):
        actual += measurement_text(case["measurement"])
    return "Pass", actual


def git_commit():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + (" (plus uncommitted changes)" if dirty else "")


def write_xlsx(rows, defects, corrected, summary):
    wb = Workbook()
    header_fill = PatternFill("solid", fgColor="DCE6F2")
    fills = {"Pass": PatternFill("solid", fgColor="E3F6EA"), "Fail": PatternFill("solid", fgColor="FDECEC"),
             "Not run": PatternFill("solid", fgColor="EEF0F3")}

    def sheet(ws, headers, data, widths):
        ws.append(headers)
        for row in data:
            ws.append(row)
        for i, width in enumerate(widths, start=1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
        for cell in ws[1]:
            cell.font, cell.fill = Font(bold=True), header_fill
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = ws.dimensions

    ws = wb.active
    ws.title = "Test cases"
    sheet(ws, COLUMNS, [[r[c] for c in COLUMNS] for r in rows], [8, 26, 12, 40, 40, 30, 40, 50, 10, 48])
    status_col = COLUMNS.index("Pass/Fail") + 1
    for row in range(2, ws.max_row + 1):
        cell = ws.cell(row=row, column=status_col)
        cell.fill, cell.font = fills.get(cell.value, PatternFill()), Font(bold=True)

    sheet(wb.create_sheet("Defects"), ["ID", "Area", "Severity", "Defect", "Found by", "Fix", "Fixed in commit"],
          [[d["id"], d["area"], d["severity"], d["defect"], d["found_by"], d["fix"], d["fixed_in"]] for d in defects],
          [6, 22, 10, 60, 34, 46, 22])
    sheet(wb.create_sheet("Corrected tests"), ["ID", "Test", "Problem", "Correction", "When"],
          [[c["id"], c["test"], c["problem"], c["correction"], c["when"]] for c in corrected],
          [6, 44, 60, 50, 22])
    sheet(wb.create_sheet("Run summary"), ["Item", "Value"], [[k, v] for k, v in summary.items()], [28, 90])
    wb.save(TESTING / "test_cases.xlsx")


def write_html(rows, defects, corrected, summary, results):
    e = html.escape
    badge = {"Pass": "pass", "Fail": "fail", "Not run": "notrun", "passed": "pass", "failed": "fail", "skipped": "notrun"}

    def table(headers, data, status_index=None):
        head = "".join(f"<th>{e(h)}</th>" for h in headers)
        body = ""
        for row in data:
            cells = ""
            for i, value in enumerate(row):
                text = e(str(value))
                if i == status_index:
                    text = f'<span class="b {badge.get(value, "")}">{text}</span>'
                cells += f"<td>{text}</td>"
            body += f"<tr>{cells}</tr>"
        return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    all_tests = [(node, outcome, f"{seconds:.2f}", message[:160])
                 for node, runs in sorted(results.items()) for outcome, seconds, message in runs]
    counts = Counter(r["Pass/Fail"] for r in rows)
    type_counts = Counter(r["Type"] for r in rows)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test report - discharge support</title>
<style>
body {{ font-family: Segoe UI, Helvetica, Arial, sans-serif; color: #1f2937; background: #fff; margin: 0; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 1.6rem; margin: 0 0 4px; }} h2 {{ font-size: 1.2rem; margin: 32px 0 8px; }}
.muted {{ color: #4b5563; font-size: 0.9rem; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
.kpi {{ border: 1px solid #dfe5ee; border-radius: 8px; padding: 10px 14px; min-width: 120px; }}
.kpi b {{ display: block; font-size: 1.4rem; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ border: 1px solid #dfe5ee; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f3f6fa; position: sticky; top: 0; }}
.b {{ border-radius: 6px; padding: 1px 8px; font-weight: 700; white-space: nowrap; }}
.pass {{ background: #e3f6ea; color: #14532d; }} .fail {{ background: #fdecec; color: #991b1b; }}
.notrun {{ background: #eef0f3; color: #374151; }}
details {{ margin-top: 8px; }}
</style></head><body><main>
<h1>Test report: AI-assisted discharge support</h1>
<p class="muted">MIS 7623 Testing Lab, mapped to the Lab 4 requirements (FR-1 to FR-7 and the non-functional requirements).
Generated {e(summary['Generated'])} from commit {e(summary['Commit'])}.</p>
<div class="kpis">
<div class="kpi"><b>{len(rows)}</b>test cases</div>
<div class="kpi"><b>{counts['Pass']}</b>passed</div>
<div class="kpi"><b>{counts['Fail']}</b>failed</div>
<div class="kpi"><b>{counts['Not run']}</b>not run</div>
<div class="kpi"><b>{type_counts['Unit']} / {type_counts['Integration']} / {type_counts['Performance']}</b>unit / integration / performance</div>
</div>
<p class="muted">Full pytest run: {e(summary['Pytest summary'])}.</p>

<h2>Test cases</h2>
{table(COLUMNS, [[r[c] for c in COLUMNS] for r in rows], status_index=COLUMNS.index('Pass/Fail'))}

<h2>Defects found during development</h2>
<p class="muted">Real defects from this project's history, with the commit that fixed each one. "Caught before commit"
means the defect was found and fixed while the stage was being built, so the fix is part of that stage's commit.</p>
{table(['ID', 'Area', 'Severity', 'Defect', 'Found by', 'Fix', 'Fixed in'],
       [[d['id'], d['area'], d['severity'], d['defect'], d['found_by'], d['fix'], d['fixed_in']] for d in defects])}

<h2>Test cases that were corrected</h2>
<p class="muted">Tests and test tooling that were wrong (or became wrong after an intended change) and how they were fixed.</p>
{table(['ID', 'Test', 'Problem', 'Correction', 'When'],
       [[c['id'], c['test'], c['problem'], c['correction'], c['when']] for c in corrected])}

<h2>Every test in this run</h2>
<details><summary>{len(all_tests)} test runs (parametrized tests count once per parameter set)</summary>
{table(['Test', 'Outcome', 'Seconds', 'Message'], all_tests, status_index=1)}
</details>

<h2>Environment</h2>
{table(['Item', 'Value'], [[k, v] for k, v in summary.items()])}
</main></body></html>"""
    (TESTING / "test_report.html").write_text(page, encoding="utf-8")


def main():
    live = "--live" in sys.argv
    pytest_summary = run_pytest(live)
    results = load_results()
    cases = json.loads((TESTING / "test_cases.json").read_text(encoding="utf-8"))["cases"]
    log = json.loads((TESTING / "defects.json").read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        status, actual = evaluate(case, results)
        rows.append({"ID": case["id"], "Requirement": case["requirement"], "Type": case["type"],
                     "Description": case["description"], "Steps": case["steps"], "Input": case["input"],
                     "Expected": case["expected"], "Actual": actual, "Pass/Fail": status,
                     "Linked pytest test": "\n".join(case["tests"])})

    summary = {
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Commit": git_commit(),
        "Pytest summary": pytest_summary,
        "Live Render test": "run" if live else "skipped (run with --live)",
        "Python": platform.python_version(),
        "Platform": platform.platform(),
    }
    write_xlsx(rows, log["defects"], log["corrected_tests"], summary)
    write_html(rows, log["defects"], log["corrected_tests"], summary, results)
    statuses = Counter(r["Pass/Fail"] for r in rows)
    print(f"pytest: {pytest_summary}")
    print(f"test cases: {dict(statuses)}  ->  testing/test_cases.xlsx, testing/test_report.html")


if __name__ == "__main__":
    main()
