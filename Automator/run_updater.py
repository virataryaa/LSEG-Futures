"""
run_updater.py — Futures (LSEG interim migration) daily updater
Called by run.bat via Task Scheduler.
Runs futures_builder_lseg.py (incremental), logs result, sends Outlook email.
Subject: [OK] ICE-FUTURES (LSEG) — YYYY-MM-DD  or  [Need Intervention] ICE-FUTURES (LSEG) — YYYY-MM-DD
"""

import subprocess
import sys
import datetime
import traceback
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oi_report import build_oi_detail, build_html_report, market_dates

ROOT     = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "Code"
DB_DIR   = ROOT / "Database"
SCRIPT   = CODE_DIR / "futures_builder_lseg.py"

COMMODITIES = ["KC", "CC", "CT", "SB", "RC", "LCC", "LSU"]
TO_EMAIL    = "virat.arya@etgworld.com"

today  = datetime.date.today()
run_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def send_outlook_email(subject: str, body: str, html: str | None = None):
    """Sends the HTML table when one is available and falls back to the plain
    text otherwise. The log keeps the plain-text version either way — run.bat
    pipes stdout through cmd, which is no place for markup."""
    try:
        import win32com.client
        outlook      = win32com.client.Dispatch("Outlook.Application")
        mail         = outlook.CreateItem(0)
        mail.To      = TO_EMAIL
        mail.Subject = subject
        if html:
            mail.HTMLBody = html
        else:
            mail.Body = body
        mail.Send()
        print(f"\n  Email sent -> {TO_EMAIL}")
    except Exception as e:
        print(f"\n  Email failed: {e}")


def row_counts() -> dict:
    counts = {}
    for comm in COMMODITIES:
        p = DB_DIR / f"{comm.lower()}_futures.parquet"
        counts[comm] = len(pd.read_parquet(p, columns=["Date"])) if p.exists() else 0
    return counts


def last_dates() -> dict:
    dates = {}
    for comm in COMMODITIES:
        p = DB_DIR / f"{comm.lower()}_futures.parquet"
        dates[comm] = pd.to_datetime(pd.read_parquet(p, columns=["Date"])["Date"]).max().date() if p.exists() else None
    return dates


print(f"\n{'='*60}")
print(f"  Futures (LSEG) Daily Update  |  {run_dt}")
print(f"{'='*60}\n")

before = row_counts()

try:
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr)
    script_ok  = result.returncode == 0
    script_err = result.stderr.strip() if result.stderr.strip() else None
except Exception as e:
    script_ok  = False
    script_err = traceback.format_exc()
    print(f"  Script launch failed: {e}")

after  = row_counts()
ldates = last_dates()
all_ok = script_ok

lines = [
    "Futures Database (LSEG interim migration) — Daily Update",
    f"Run time  : {run_dt}",
    "",
    f"{'COMMODITY':<10} {'ROWS UPSERTED':>14}  {'TOTAL ROWS':>12}  "
    f"{'LAST DATE':>12}  {'OI DATE':>12}  STATUS",
    "-" * 78,
]
oi_lagging, summary_rows = [], []
for comm in COMMODITIES:
    upserted   = after[comm] - before[comm]
    total      = after[comm]
    last_d     = str(ldates[comm]) if ldates[comm] else "N/A"
    px_date, oi_date = market_dates(DB_DIR, comm)
    oi_d       = str(oi_date.date()) if oi_date is not None else "N/A"
    # A session's settlement lands before its open interest, so "OK" against the
    # settlement date alone would read as if the OI were current too.
    lag        = px_date is not None and oi_date is not None and px_date > oi_date
    if lag:
        oi_lagging.append(comm)
    status_str = "MISSING" if total == 0 else ("OI LAGS" if lag else "OK")
    lines.append(f"{comm:<10} {upserted:>14,}  {total:>12,}  {last_d:>12}  "
                  f"{oi_d:>12}  {status_str}")
    summary_rows.append(dict(comm=comm, upserted=upserted, total=total,
                             last_date=last_d, oi_date=oi_d, status=status_str))

if oi_lagging:
    lines += ["",
              f"NOTE: {', '.join(oi_lagging)} carry a settlement for a session whose open",
              "      interest the vendor has not published yet. The builder's quote top-up",
              "      fills it on the next run once the exchange publishes."]

if script_err:
    label = "--- ERRORS ---" if not script_ok else "--- STDERR (non-fatal, script exit code 0) ---"
    lines += ["", label, script_err]

lines += ["", "=" * 78,
          "PER-CONTRACT DETAIL (latest open-interest date per market)",
          "=" * 78, ""]
try:
    lines.append(build_oi_detail(DB_DIR, COMMODITIES))
except Exception:
    lines += ["  detail section failed:", traceback.format_exc()]

body    = "\n".join(lines)
try:
    html_body = build_html_report(DB_DIR, COMMODITIES, run_dt, summary_rows)
except Exception:
    html_body = None
    print("  HTML report failed, falling back to plain text:")
    print(traceback.format_exc())
subject = f"[OK] ICE-FUTURES (LSEG) — {today}" if all_ok else f"[Need Intervention] ICE-FUTURES (LSEG) — {today}"

print(f"\n{body}")

if script_ok:
    try:
        subprocess.run(["git", "-C", str(ROOT), "add", "Database/"], capture_output=True, text=True)
        subprocess.run(["git", "-C", str(ROOT), "commit", "-m", f"auto: futures update (LSEG) {today}"],
                        capture_output=True, text=True)
        push_result = subprocess.run(["git", "-C", str(ROOT), "push"], capture_output=True, text=True)
        if push_result.returncode == 0:
            print(f"\n  Git push OK")
        else:
            print(f"\n  Git push failed: {push_result.stderr.strip()}")
    except Exception as e:
        print(f"\n  Git push error: {e}")

send_outlook_email(subject, body, html_body)
print(f"\nDone — {run_dt}")
