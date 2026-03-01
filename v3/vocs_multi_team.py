"""
Voice of Customer Synthesizer V3 - Multi-Team Edition
Analyzes feedback for multiple teams with WoW trending
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
from anthropic import Anthropic

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

VERSION = "3.0"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
CONFIG_FILE = SCRIPT_DIR / "teams_config.json"
PROMPT_TEMPLATE_FILE = SCRIPT_DIR / "team_feedback_prompt_template.txt"
DB_PATH = SCRIPT_DIR / "vocs_history.db"

# Column name detection
DATE_COLUMNS = ["date", "submitted"]
SESSION_COLUMNS = ["session_id", "sessionid"]
MAX_ROWS_FOR_CLAUDE = 500

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096


def load_config() -> Dict:
    """Load teams configuration"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_template() -> str:
    """Load the team feedback prompt template"""
    if not PROMPT_TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_TEMPLATE_FILE}")
    return PROMPT_TEMPLATE_FILE.read_text(encoding='utf-8')


def find_team_csv(team_config: Dict) -> Optional[Path]:
    """Find the latest CSV file matching team's pattern"""
    pattern = team_config['csv_pattern']
    candidates = list(INPUT_DIR.glob(pattern))
    if not candidates:
        print(f"  ⚠️  No CSV found matching: {pattern}")
        return None
    # Return most recent
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column name that exists (case-insensitive)"""
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

def load_and_prepare_feedback(csv_path: Path, team_config: Dict) -> tuple[pd.DataFrame, Dict]:
    """Load CSV and prepare feedback data with metadata"""
    df = pd.read_csv(csv_path)
    
    # Get column mappings from team config
    col_map = team_config.get('columns', {})
    
    session_col = col_map.get('session_id')
    ces_col = col_map.get('ces_score')
    date_col = col_map.get('date')
    feedback_col_1 = col_map.get('feedback_1')
    feedback_col_2 = col_map.get('feedback_2')
    feedback_col_3 = col_map.get('feedback_3')
    
    # Verify required columns exist
    missing = []
    if session_col not in df.columns:
        missing.append(f"session_id ({session_col})")
    if ces_col not in df.columns:
        missing.append(f"ces_score ({ces_col})")
    if date_col not in df.columns:
        missing.append(f"date ({date_col})")
    if feedback_col_1 not in df.columns and feedback_col_2 not in df.columns:
        missing.append("feedback columns")
    
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {', '.join(missing)}")
    
    # Combine feedback columns (support up to 3)
    df = df.copy()
    feedback_parts = []
    
    if feedback_col_1 and feedback_col_1 in df.columns:
        df['_col1_'] = df[feedback_col_1].fillna('').astype(str).str.strip()
        feedback_parts.append(df['_col1_'].apply(lambda x: f"[Reason 1]: {x}" if x else ""))
    
    if feedback_col_2 and feedback_col_2 in df.columns:
        df['_col2_'] = df[feedback_col_2].fillna('').astype(str).str.strip()
        feedback_parts.append(df['_col2_'].apply(lambda x: f"[Reason 2]: {x}" if x else ""))
    
    if feedback_col_3 and feedback_col_3 in df.columns:
        df['_col3_'] = df[feedback_col_3].fillna('').astype(str).str.strip()
        feedback_parts.append(df['_col3_'].apply(lambda x: f"[Improvement]: {x}" if x else ""))
    
    # Combine all feedback parts
    combined_series = []
    for part in feedback_parts:
        combined_series.append(part)
    
    if len(combined_series) > 0:
        # Combine with |, filtering out empty strings
        df['_combined_feedback_'] = combined_series[0]
        for i in range(1, len(combined_series)):
            df['_combined_feedback_'] = df['_combined_feedback_'] + " | " + combined_series[i]
        df['_combined_feedback_'] = df['_combined_feedback_'].str.strip(' | ')
    else:
        df['_combined_feedback_'] = ""
    
    # Standardize column names for rest of pipeline
    df['_session_id_'] = df[session_col]
    df['_ces_score_'] = df[ces_col]
    df['_date_'] = df[date_col]
    
    # Filter to rows with feedback
    df_with_text = df[df['_combined_feedback_'] != ""].copy()
    
    # Calculate CES breakdown (scores 1-5)
    ces_breakdown = {}
    if '_ces_score_' in df_with_text.columns:
        ces_breakdown = df_with_text['_ces_score_'].value_counts().to_dict()
    
    metadata = {
        'total_rows': len(df),
        'rows_with_text': len(df_with_text),
        'ces_breakdown': ces_breakdown,
        'team_name': team_config['name'],
        'team_id': team_config['id']
    }
    
    return df_with_text, metadata

def build_feedback_summary(df: pd.DataFrame, metadata: Dict, max_rows: int = 500) -> str:
    """Format feedback for Claude"""
    use_cols = ['_ces_score_', '_session_id_', '_date_', '_combined_feedback_']
    df_show = df[use_cols].head(max_rows)
    
    lines = []
    for idx, (_, row) in enumerate(df_show.iterrows(), start=1):
        parts = [
            f"ces_score={row['_ces_score_']}",
            f"session_id={row['_session_id_']}",
            f"date={row['_date_']}",
            f"feedback={row['_combined_feedback_']}"
        ]
        lines.append(f"[{idx}] " + " | ".join(parts))
    
    formatted = "\n".join(lines)
    if len(df) > max_rows:
        formatted += f"\n\n... ({len(df) - max_rows} more rows omitted)"
    
    return formatted

def get_week_number() -> str:
    """Get current week in YYYY-WXX format"""
    now = datetime.now()
    week_num = now.isocalendar()[1]
    return f"{now.year}-W{week_num:02d}"


def get_last_week_data(team_id: str) -> Optional[Dict]:
    """Retrieve last week's metrics for this team from SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT week_date, total_feedback, ces_avg, negative_count, analysis_text
            FROM team_summaries
            WHERE team_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (team_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'week_date': row[0],
                'total_feedback': row[1],
                'ces_avg': row[2],
                'negative_count': row[3],
                'analysis_text': row[4]
            }
        return None
    except Exception as e:
        print(f"  Warning: Could not fetch last week's data: {e}")
        return None

def customize_prompt(template: str, team_config: Dict, metadata: Dict, week_number: str, last_week_data: Optional[Dict]) -> str:
    """Fill in template with team-specific data and trends"""
    ces_breakdown = metadata.get('ces_breakdown', {})
    
    # Calculate WoW trends
    wow_trends = ""
    if last_week_data:
        total_change = metadata['rows_with_text'] - last_week_data['total_feedback']
        total_pct = (total_change / last_week_data['total_feedback'] * 100) if last_week_data['total_feedback'] > 0 else 0
        total_arrow = "↑" if total_change > 0 else "↓" if total_change < 0 else "→"
        
        ces_change = metadata.get('ces_avg', 0) - last_week_data.get('ces_avg', 0)
        ces_arrow = "↓" if ces_change < 0 else "↑" if ces_change > 0 else "→"  # Note: lower CES is worse
        
        neg_change = metadata.get('negative_count', 0) - last_week_data.get('negative_count', 0)
        neg_pct = (neg_change / last_week_data['negative_count'] * 100) if last_week_data.get('negative_count', 0) > 0 else 0
        neg_arrow = "↑" if neg_change > 0 else "↓" if neg_change < 0 else "→"
        
        wow_trends = f"""
WEEK-OVER-WEEK COMPARISON (vs {last_week_data['week_date']}):
- Total feedback: {total_arrow} {abs(total_pct):.0f}% ({last_week_data['total_feedback']} → {metadata['rows_with_text']})
- Average CES: {ces_arrow} {abs(ces_change):.2f} ({last_week_data.get('ces_avg', 0):.2f} → {metadata.get('ces_avg', 0):.2f})
- Negative scores (1-2): {neg_arrow} {abs(neg_pct):.0f}% ({last_week_data.get('negative_count', 0)} → {metadata.get('negative_count', 0)})
"""
    else:
        wow_trends = "\nFIRST WEEK - No previous data for comparison\n"
    
    # Add explicit context at the top
    context_header = f"""
IMPORTANT CONTEXT:
- Product Area: {team_config['name']}
- Week: {week_number}
- Total Feedback Entries: {metadata['rows_with_text']}
- Data includes: Session IDs, CES scores (1-5), dates, and customer comments

{wow_trends}
"""
    
    prompt = context_header + template.format(
        team_name=team_config['name'],
        market=team_config['market'],
        team_context=team_config['context'],
        priority_areas=", ".join(team_config['priority_areas']),
        week_number=week_number,
        total_count=metadata['rows_with_text'],
        ces_5=ces_breakdown.get(5, 0),
        ces_4=ces_breakdown.get(4, 0),
        ces_3=ces_breakdown.get(3, 0),
        ces_2=ces_breakdown.get(2, 0),
        ces_1=ces_breakdown.get(1, 0)
    )
    
    return prompt

def analyze_with_claude(prompt: str, feedback_summary: str) -> str:
    """Send to Claude for analysis"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = Anthropic(api_key=api_key)
    
    user_content = f"{prompt}\n\n---\nFEEDBACK DATA:\n\n{feedback_summary}"
    
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": user_content}]
    )
    
    if not response.content or response.content[0].type != "text":
        raise RuntimeError("Claude returned unexpected content")
    
    return response.content[0].text

def post_to_slack(webhook_url: str, text: str, channel: str) -> None:
    """Post message to Slack"""
    print(f"    DEBUG: Webhook URL: {webhook_url[:50]}...")
    print(f"    DEBUG: Target channel: {channel}")
    print(f"    DEBUG: Message length: {len(text)} chars")
    
    payload = {"text": text}
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        resp = requests.post(webhook_url, json=payload, timeout=30, verify=False)
        print(f"    DEBUG: Response status: {resp.status_code}")
        print(f"    DEBUG: Response body: {resp.text}")
        resp.raise_for_status()
        print(f"    ✅ Posted successfully")
    except Exception as e:
        print(f"    ❌ Slack error: {e}")
        raise

def init_db() -> None:
    """Create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Team summaries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            team_name TEXT NOT NULL,
            week_date TEXT NOT NULL,
            total_feedback INTEGER NOT NULL,
            ces_avg REAL,
            negative_count INTEGER,
            analysis_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Run log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            teams_processed INTEGER NOT NULL,
            teams_failed INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()


def save_team_summary(team_id: str, team_name: str, week_date: str, 
                     total_feedback: int, ces_avg: float, negative_count: int, 
                     analysis_text: str) -> None:
    """Save team's weekly summary to database"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO team_summaries 
        (team_id, team_name, week_date, total_feedback, ces_avg, negative_count, analysis_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (team_id, team_name, week_date, total_feedback, ces_avg, negative_count, analysis_text))
    
    conn.commit()
    conn.close()

def process_team(team_config: Dict, prompt_template: str, week_number: str) -> bool:
    """Process a single team's feedback"""
    team_id = team_config['id']
    team_name = team_config['name']
    
    print(f"\n{'='*60}")
    print(f"Processing: {team_name}")
    print(f"{'='*60}")
    
    # Get team-specific webhook
    webhook_env_var = team_config.get('slack_webhook_env')
    if not webhook_env_var:
        print(f"  ⚠️  No slack_webhook_env defined for {team_name}")
        return False
    
    slack_webhook = os.environ.get(webhook_env_var)
    if not slack_webhook:
        print(f"  ⚠️  {webhook_env_var} not set in environment")
        return False
    
    # Find CSV
    csv_path = find_team_csv(team_config)
    if not csv_path:
        return False
    
    print(f"  📄 CSV: {csv_path.name}")
    
    # Load and prepare data
    df, metadata = load_and_prepare_feedback(csv_path, team_config)
    print(f"  📊 Loaded: {metadata['rows_with_text']} feedback entries with text")
    
    # Calculate metrics for storage
    ces_avg = df['_ces_score_'].mean() if '_ces_score_' in df.columns else 0.0
    negative_count = len(df[df['_ces_score_'] <= 2]) if '_ces_score_' in df.columns else 0
    
    # Store metrics in metadata for prompt
    metadata['ces_avg'] = ces_avg
    metadata['negative_count'] = negative_count
    
    # Get last week's data for comparison
    last_week_data = get_last_week_data(team_id)
    if last_week_data:
        print(f"  📈 Comparing to last week ({last_week_data['week_date']})")
    
    # Build feedback summary
    feedback_summary = build_feedback_summary(df, metadata, MAX_ROWS_FOR_CLAUDE)
    
    # Customize prompt with WoW trends
    prompt = customize_prompt(prompt_template, team_config, metadata, week_number, last_week_data)
    
    # Analyze with Claude
    print(f"  🤖 Calling Claude API...")
    analysis = analyze_with_claude(prompt, feedback_summary)
    print(f"  ✅ Analysis complete ({len(analysis)} characters)")
    
    # Post to Slack
    channel = team_config['slack_channel']
    print(f"  📤 Posting to Slack: {channel}")
    post_to_slack(slack_webhook, analysis, channel)
    
    # Save to database
    save_team_summary(team_id, team_name, week_number, metadata['rows_with_text'], 
                     ces_avg, negative_count, analysis)
    print(f"  💾 Saved to database")
    
    return True

def main() -> int:
    """Main execution"""
    print(f"{'='*60}")
    print(f"VoCS Multi-Team Analyzer v{VERSION}")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        config = load_config()
        teams = config['teams']
        global_settings = config['global_settings']
        
        print(f"Loaded {len(teams)} teams from config")
        
        # Load prompt template
        prompt_template = load_prompt_template()
        print(f"Loaded prompt template ({len(prompt_template)} chars)")
        
        # Get current week
        week_number = get_week_number()
        print(f"Processing week: {week_number}\n")
        
        # Process each team
        success_count = 0
        fail_count = 0
        
        for team in teams:
            try:
                if process_team(team, prompt_template, week_number):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  ❌ Error processing {team['name']}: {e}")
                fail_count += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {success_count}/{len(teams)}")
        print(f"❌ Failed: {fail_count}/{len(teams)}")
        
        return 0 if fail_count == 0 else 1
        
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())