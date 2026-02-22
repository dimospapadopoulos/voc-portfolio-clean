"""
Weekly feedback analyser (V2): reads weekly signal data (CSV or Excel) and prompt,
sends to Claude for analysis, posts to Slack, and stores summaries in SQLite.
Analyzes ALL feedback with text (no signal_score filtering).
"""
from dotenv import load_dotenv

import os
load_dotenv()
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import requests
from anthropic import Anthropic

VERSION = "2.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
PROMPT_FILE = SCRIPT_DIR / "weekly_feedback_prompt.txt"
DB_PATH = SCRIPT_DIR / "vocs_history.db"
# weekly_signal_YYYY-WXX where YYYY=year, XX=week number; supports .csv and .xlsx
WEEKLY_SIGNAL_CSV_GLOB = "weekly_signal_*.csv"
WEEKLY_SIGNAL_XLSX_GLOB = "weekly_signal_*.xlsx"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096

# Column name variants (first match wins)
COMMENT_COLUMNS = ["feedback_text", "comment_text", "comment", "feedback"]
DATE_COLUMNS = ["date_received", "date", "received_at"]
MAX_ROWS_FOR_CLAUDE = 500


def find_latest_weekly_signal() -> Path:
    """Find the most recent weekly_signal_YYYY-WXX.csv or .xlsx under input folder."""
    candidates = (
        list(INPUT_DIR.rglob(WEEKLY_SIGNAL_CSV_GLOB))
        + list(INPUT_DIR.rglob(WEEKLY_SIGNAL_XLSX_GLOB))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No file matching '{WEEKLY_SIGNAL_CSV_GLOB}' or '{WEEKLY_SIGNAL_XLSX_GLOB}' "
            f"found under {INPUT_DIR}"
        )
    # Sort by modification time, newest first
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def load_signal_file(path: Path) -> pd.DataFrame:
    """Load weekly signal data from CSV or Excel; return DataFrame."""
    path_str = str(path).lower()
    if path_str.endswith(".csv"):
        print(f"Reading CSV: {path}")
        df = pd.read_csv(path)
    elif path_str.endswith(".xlsx"):
        print(f"Reading Excel: {path}")
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .xlsx")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name that exists in df (case-insensitive match)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def build_data_summary_for_claude(df: pd.DataFrame) -> tuple[str, int]:
    """
    Build a clean list of feedback rows for Claude: only rows with non-empty comment.
    Columns included: ces_score, session_id, date_received, comment_text (or detected names).
    Returns (formatted_string, total_rows_with_comment).
    """
    comment_col = _resolve_column(df, COMMENT_COLUMNS)
    date_col = _resolve_column(df, DATE_COLUMNS)
    if not comment_col:
        raise ValueError(
            f"No comment/feedback column found. Looked for: {COMMENT_COLUMNS}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.copy()
    df["_comment_"] = df[comment_col].astype(str).str.strip()
    df_with_text = df[df["_comment_"] != ""].copy()
    total = len(df_with_text)
    if total == 0:
        return "No feedback with text in this dataset.", 0

    use_cols = [c for c in ["ces_score", "session_id", date_col, comment_col] if c and c in df_with_text.columns]
    df_show = df_with_text[use_cols].head(MAX_ROWS_FOR_CLAUDE)
    lines = []
    for j, (_, row) in enumerate(df_show.iterrows(), start=1):
        parts = []
        if "ces_score" in row:
            parts.append(f"ces_score={row['ces_score']}")
        if "session_id" in row:
            parts.append(f"session_id={row['session_id']}")
        if date_col:
            parts.append(f"date_received={row[date_col]}")
        parts.append(f"comment_text={row[comment_col]}")
        lines.append(f"[{j}] " + " | ".join(str(p) for p in parts))
    formatted = "\n".join(lines)
    if total > MAX_ROWS_FOR_CLAUDE:
        formatted += f"\n\n... ({total - MAX_ROWS_FOR_CLAUDE} more rows omitted)"
    return formatted, total


def load_prompt() -> str:
    """Load the weekly feedback prompt from text file."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    print(f"Reading prompt from: {PROMPT_FILE}")
    text = PROMPT_FILE.read_text(encoding="utf-8")
    print(f"  Loaded {len(text)} characters.")
    return text.strip()


def get_anthropic_client() -> Anthropic:
    """Create Anthropic client using API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with your Claude API key."
        )
    return Anthropic(api_key=api_key)


def init_db() -> None:
    """Create summaries table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_date TEXT NOT NULL,
                total_feedback INTEGER NOT NULL,
                analysis_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_summary(week_date: str, total_feedback: int, analysis_text: str) -> None:
    """Insert one week's summary into vocs_history.db."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO summaries (week_date, total_feedback, analysis_text, created_at) VALUES (?, ?, ?, datetime('now'))",
            (week_date, total_feedback, analysis_text),
        )
    print(f"  Saved summary to {DB_PATH} (week_date={week_date}).")


def analyze_with_claude(prompt: str, data_summary: str) -> str:
    """Send prompt and data to Claude and return the analysis text."""
    client = get_anthropic_client()
    user_content = (
        f"{prompt}\n\n"
        "---\n"
        "Data to analyze (all feedback with text):\n\n"
        f"{data_summary}"
    )
    print("Calling Claude API...")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": user_content}],
    )
    # Extract text from the first content block
    if not response.content:
        raise RuntimeError("Claude returned no content.")
    block = response.content[0]
    if block.type != "text":
        raise RuntimeError(f"Unexpected content type: {block.type}")
    text = block.text
    print(f"  Received {len(text)} characters from Claude.")
    return text


def post_to_slack(webhook_url: str, text: str) -> None:
    """Post message to Slack via webhook."""
    print("Posting to Slack...")
    payload = {"text": text}
    resp = requests.post(webhook_url, json=payload, timeout=30)
    resp.raise_for_status()
    print("  Slack post successful.")


def get_week_date_from_path(path: Path) -> str:
    """Extract YYYY-WXX from filename like weekly_signal_2025-W52.csv."""
    stem = path.stem  # e.g. weekly_signal_2025-W52
    if "weekly_signal_" in stem:
        return stem.replace("weekly_signal_", "")
    return stem


def main() -> int:
    """Run the full pipeline: signal file (CSV/Excel) -> prompt -> Claude -> Slack -> SQLite."""
    print("=" * 60)
    print(f"Weekly Feedback Analyser v{VERSION}")
    print("=" * 60)

    try:
        # 1. Find and load weekly signal file (CSV or Excel)
        signal_path = find_latest_weekly_signal()
        df = load_signal_file(signal_path)
        week_date = get_week_date_from_path(signal_path)

        # 2. Build data summary: all rows with non-empty comment; ces_score, session_id, date_received, comment_text
        data_summary, total_feedback = build_data_summary_for_claude(df)
        print(f"  Sending {min(total_feedback, MAX_ROWS_FOR_CLAUDE)} rows to Claude (total with text: {total_feedback}).")

        # 3. Load prompt
        prompt = load_prompt()

        # 4. Call Claude
        analysis = analyze_with_claude(prompt, data_summary)

        # 5. Slack webhook (unchanged)
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook_url:
            print("SLACK_WEBHOOK_URL not set. Skipping Slack post.")
            print("Analysis preview:")
            print("-" * 40)
            print(analysis[:1500] + ("..." if len(analysis) > 1500 else ""))
            save_summary(week_date, total_feedback, analysis)
            return 0
        post_to_slack(webhook_url, analysis)

        # 6. Store summary in SQLite
        save_summary(week_date, total_feedback, analysis)

        print("=" * 60)
        print("Done.")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
