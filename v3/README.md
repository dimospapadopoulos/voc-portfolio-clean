# V3 - Production Multi-Team Platform

## What's New in V3

V2 served one team. V3 is a **multi-tenant platform** serving 7 product teams with automated scheduling.

### Key Features

**🏢 Multi-Team Architecture**
- 7 teams: Checkout (UK/US), Catalogue (UK/US), Account (UK/US), DAS (UK)
- Team-specific configurations, prompts, and Slack channels
- Isolated data pipelines with shared infrastructure
- Support for different CSV schemas per team

**📊 Week-over-Week Trend Tracking**
- Automatic comparison to previous week's metrics
- Delta indicators: ↑ volume up, ↓ CES score down, → stable
- Historical data stored in SQLite
- Trend context included in AI analysis

**⏰ Scheduled Automation**
- Windows Task Scheduler integration
- Runs every Monday 10am GMT automatically
- Batch script with environment variable loading
- Error retry logic (3 attempts)

**💬 Enhanced Slack Integration**
- 4 separate webhooks for different team channels
- Team-specific routing based on config
- Formatted messages with emojis and structure
- Session IDs linked to video replay tools

**🗄️ Production Database**
- SQLite historical storage
- Team-level metrics tracking
- Week-by-week comparison queries
- Foundation for future analytics dashboard

## Architecture
```
Sunday: Manual CSV download from Mouseflow (7 files)
    ↓
Monday 10am: Task Scheduler triggers run_vocs.bat
    ↓
Multi-Team Analyzer:
  ├─ Load teams_config.json (7 team configs)
  ├─ For each team:
  │   ├─ Load CSV with team-specific column mapping
  │   ├─ Combine up to 3 feedback columns
  │   ├─ Fetch last week's data from SQLite
  │   ├─ Send to Claude with WoW context
  │   ├─ Post analysis to team's Slack channel
  │   └─ Save metrics to database
  └─ Report: 7/7 teams successful
```

## Tech Stack

**Core:**
- Python 3.11
- Anthropic Claude Sonnet 4.5 API
- Pandas (data processing)
- SQLite (historical storage)

**Integrations:**
- Slack Webhooks API (4 channels)
- Mouseflow (CSV export, manual for V3)

**Automation:**
- Windows Task Scheduler
- Batch scripting
- Environment variable management

## Configuration System

### Team Config Structure

Each team has:
- **csv_pattern**: Glob pattern to find their data file
- **columns**: Mapping of logical names to actual CSV columns
- **slack_channel**: Destination for analysis
- **slack_webhook_env**: Environment variable with webhook URL
- **context**: Business context for AI analysis
- **priority_areas**: Focus topics for categorization

This allows **zero code changes** when:
- Adding new teams
- Changing CSV schemas
- Updating business priorities
- Switching Slack channels

### Column Flexibility

Handles:
- Different CES question text across teams
- 2 or 3 feedback columns (automatically combined)
- Varying date/session ID column names
- Market-specific survey variations

Example:
```python
# Catalogue has 3 feedback columns
"feedback_1": "Reason for score",
"feedback_2": "Additional feedback", 
"feedback_3": "How to improve"

# Checkout has 2 feedback columns  
"feedback_1": "Reason for score",
"feedback_2": "How to improve"
```

System handles both automatically.

## Database Schema
```sql
-- Team weekly summaries
CREATE TABLE team_summaries (
  id INTEGER PRIMARY KEY,
  team_id TEXT NOT NULL,
  team_name TEXT NOT NULL,
  week_date TEXT NOT NULL,        -- e.g., "2026-W10"
  total_feedback INTEGER NOT NULL,
  ces_avg REAL,
  negative_count INTEGER,         -- CES scores 1-2
  analysis_text TEXT NOT NULL,    -- Full Slack message
  created_at TEXT NOT NULL
);

-- Run audit log
CREATE TABLE run_log (
  id INTEGER PRIMARY KEY,
  run_date TEXT NOT NULL,
  teams_processed INTEGER NOT NULL,
  teams_failed INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
```

## Performance & Scale

**Current Load:**
- 7 teams × ~150 feedback entries/team = 1,050 entries/week
- Runtime: 3-5 minutes total
- Claude API: ~$0.40-0.60 per run
- Monthly cost: ~$2-3

**Scalability:**
- Can support 20+ teams with same infrastructure
- Parallel processing possible (not implemented yet)
- Rate limiting handled gracefully
- Database indexed for fast trend queries

## Sample Output

**Slack Message Format:**
```
*CES - Checkout (UK) - WEEK W10 SUMMARY* 📊

WEEK-OVER-WEEK COMPARISON (vs 2026-W09):
- Total feedback: ↑ 12% (523 → 586)
- Average CES: ↓ 0.3 (4.1 → 3.8)
- Negative scores (1-2): ↑ 18% (198 → 234) ⚠️

Total: 586 | CES: 5:234 | 4:156 | 3:98 | 2:67 | 1:31

🔴 Payment failures after 3D Secure (Severity: 9/10)
   • Impact: 47 customers
   • Sessions: sess_abc123, sess_def456, sess_ghi789
   • Trend: NEW this week

🟡 Slow mobile checkout (Severity: 6/10)
   • Impact: 23 customers
   • Sessions: sess_jkl012, sess_mno345
   • Trend: ↑ 40% vs last week

*Recommended Actions:*
1. Check payment gateway logs for 3D Secure errors
2. Run mobile performance audit
3. Add progress indicators to reduce perceived wait
```

## V1 → V2 → V3 Evolution

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| Manual weighting | Excel formulas | None | None |
| AI analysis | Keywords only | Full feedback | Full + trends |
| Teams | 1 | 1 | 7 |
| Slack channels | 1 | 1 | 4 |
| Automation | Manual | Manual | Scheduled |
| Trend tracking | None | None | WoW deltas |
| Database | None | Basic | Production |
| Column mapping | Hardcoded | Hardcoded | Configurable |
| Setup time | 2 hours/week | 10 min/week | 5 min/week |

## Deployment

**Requirements:**
- Python 3.11+
- Windows 10+ (for Task Scheduler)
- Anthropic API key
- Slack webhook URLs (one per channel)

**Setup:**
1. Install dependencies: `pip install pandas anthropic requests python-dotenv openpyxl urllib3`
2. Configure `teams_config.json` with your teams
3. Set environment variables (API keys, webhooks)
4. Test: `python vocs_multi_team.py`
5. Schedule: Import `run_vocs.bat` into Task Scheduler

**Weekly Workflow:**
1. **Sunday night:** Download CSVs from Mouseflow → save to `input/` folder
2. **Monday 10am:** Task runs automatically
3. **Check:** Slack channels for summaries

## What I Learned Building V3

**Product Management:**
- Multi-stakeholder requirements gathering (7 teams)
- Designing for extensibility (config-driven architecture)
- Balancing automation vs. flexibility
- ROI measurement (time saved vs. infrastructure cost)

**Technical:**
- Multi-tenant system design
- Database schema for time-series analytics
- Error handling and retry logic
- Windows automation and batch scripting
- Environment variable security patterns
- CSV schema normalization strategies

**Iteration Strategy:**
- V1: Prove value with manual PoC (1 team)
- V2: Automate core workflow (still 1 team)
- V3: Scale to production (7 teams, scheduling)
- V4: Full automation (Mouseflow API integration)

This staged approach de-risked each step and maintained team confidence.

## Roadmap

- ✅ V1: Manual keyword weighting PoC and Claude limited pilot (1 team)
- ✅ V2: AI-powered automation (1 team)  
- ✅ V3: Multi-team platform with scheduling (7 teams)
- 🔄 V4: Mouseflow API integration (eliminate manual download)
- 📋 V4: Email summaries with trend charts
- 📋 V4: GitHub Actions (cloud-based scheduling)
- 📋 V5: Predictive issue detection (ML on historical trends)
- 📋 V5: Cross-team pattern analysis

## Real-World Impact

**Quantified Results:**
- **Time saved:** 8-12 hours/month across 7 inputs
- **Market:** Scalable for multiple markets
- **Consistency:** Every team gets same analysis quality
- **Speed:** Feedback → insights in 3 minutes vs. 2+ hours weekly
- **Visibility:** Historical trends now trackable
- **Cost:** $2-3/month vs. hours of PM/analyst time

**Team Feedback:**
- "Finally see patterns we were missing manually"
- "Are you turning into an engineer?"
- "Awesome!! How can we automate it even further?"

This tool will be now part of weekly product review meetings.