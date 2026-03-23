# VoCS V4 - Voice of Customer Synthesizer

**Production-grade AI-powered customer feedback analysis system with automated testing, validation, and monitoring.**

## Overview

VoCS V4 is a multi-team feedback analysis tool that processes 1,000+ weekly customer survey responses, identifies critical issues, and posts automated summaries to Slack. Built entirely by a non-technical Product Manager using Claude API + Python.

### Key Features

- **Multi-team support**: Analyze 7+ product teams with separate Slack channels
- **Week-over-week trending**: Track CES scores and issue volume changes
- **Automated scheduling**: Windows Task Scheduler for hands-free weekly execution
- **Data validation**: Pre-flight checks before analysis
- **Dry-run mode**: Test without posting to Slack
- **Regression testing**: Golden test set evaluation with 96% severity accuracy
- **Historical database**: SQLite storage for long-term trend analysis

## Architecture

```
Weekly Workflow:
Sunday: Download CSVs from analytics platform → input/
Monday 10am: Task Scheduler triggers run_vocs.bat
  ├─ Validator checks CSV integrity
  ├─ Main analyzer processes each team
  │   ├─ Load CSV (1000 rows: 500 with comments, 500 score-only)
  │   ├─ Calculate metrics for ALL rows
  │   ├─ Analyze feedback with Claude API
  │   ├─ Post summary to team Slack channel
  │   └─ Save to historical database
  └─ Generate weekly report
```

## Tech Stack

- **Python 3.12+**: Core language
- **Claude API**: Sonnet 4.5 for analysis
- **Pandas**: CSV processing
- **SQLite**: Historical storage
- **Slack Webhooks**: Team notifications
- **Windows Task Scheduler**: Automation

## Installation

### Prerequisites

```powershell
# Install dependencies
pip install pandas anthropic requests python-dotenv urllib3
```

### Configuration

1. **Copy environment template:**
```powershell
copy .env.example .env
```

2. **Add your credentials** to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
SLACK_WEBHOOK_CHECKOUT=https://hooks.slack.com/services/...
```

3. **Configure teams** in `teams_config.json`:
   - Update team IDs, names, markets
   - Set CSV file patterns
   - Map column names to your survey
   - Add product context and priority areas

4. **Customize prompt** in `team_feedback_prompt_template.txt`:
   - Add company-specific terminology
   - Adjust tone and formatting
   - Define output structure

## Usage

### Manual Run

```powershell
# Run all teams
python vocs_multi_team.py

# Dry-run (no Slack posting)
python vocs_multi_team.py --dry-run

# Test mode (no Slack, no database)
python vocs_multi_team.py --test
```

### Validation

```powershell
# Validate CSVs before running
python vocs_validator.py
```

### Database Queries

```powershell
# View historical trends
python check_database.py
```

### Automated Scheduling

1. Copy `run_vocs.bat.example` to `run_vocs.bat`
2. Update paths in the batch file
3. Create Windows Task Scheduler task:
   - Trigger: Weekly, Monday, 10:00 AM
   - Action: Run `run_vocs.bat`
   - Setting: "Run only when user is logged on"

## Evaluation Framework

### Golden Test Sets

50 hand-labeled examples across teams for regression testing:

```json
{
  "id": "CHK_UK_001",
  "ces_score": 1,
  "feedback": "[Reason 1]: Payment failed repeatedly",
  "expected": {
    "severity": 10,
    "category": "Payment",
    "top_3": true
  }
}
```

### Run Evaluations

```powershell
# Test all teams
python tests/eval_runner.py

# Test specific team
python tests/eval_runner.py checkout_uk

# View accuracy trends
python tests/view_trends.py
```

### Accuracy Metrics

- **Category**: 80%+ (Payment, Mobile, UX, Technical, etc.)
- **Severity**: 96%+ (0-10 impact scale)
- **Top-3**: 66%+ (Critical issues in Top 3 section)

## File Structure

```
vocs_v4/
├── vocs_multi_team.py              # Main analyzer
├── vocs_validator.py               # Pre-flight validation
├── check_database.py               # Database viewer
├── teams_config.json               # Team configurations
├── team_feedback_prompt_template.txt  # Analysis prompt
├── .env.example                    # Environment template
├── run_vocs.bat.example            # Scheduler template
├── vocs_history.db                 # SQLite database (generated)
├── input/                          # Weekly CSV files
├── tests/
│   ├── eval_runner.py              # Regression testing
│   ├── golden_test_set.json        # Test examples
│   ├── view_trends.py              # Accuracy trends
│   ├── debug_eval.py               # Debug failures
│   └── eval_results/               # Test outputs
└── README.md                       # This file
```

## Key Improvements Over V3

### Data Integrity
- ✅ Counts ALL survey responses (not just ones with comments)
- ✅ CES breakdown includes score-only responses
- ✅ Tracks comment rate (e.g., 500/1000 = 50%)
- ✅ Week-over-week trends use correct totals

### Testing & Validation
- ✅ Batch evaluation mode (matches production workflow)
- ✅ 50-example golden test set
- ✅ Accuracy tracking over time
- ✅ CSV validation before analysis

### Operational
- ✅ Dry-run mode for safe testing
- ✅ Database viewer for historical queries
- ✅ Better error handling
- ✅ Corporate SSL bypass

## Prompt Engineering

The system uses a dynamic prompt template with team-specific context:

**Placeholders:**
- `{team_name}`, `{market}`, `{week_number}`
- `{total_count}`, `{comments_count}`
- `{ces_1}`, `{ces_2}`, `{ces_3}`, `{ces_4}`, `{ces_5}`
- `{team_context}`, `{priority_areas}`

**Output Structure:**
- Executive summary (1-3 sentences)
- Top 3 critical issues (with session IDs)
- Other issues (grouped by theme)
- Positive highlights

## Database Schema

```sql
-- Team summaries
CREATE TABLE team_summaries (
    id INTEGER PRIMARY KEY,
    team_id TEXT,
    team_name TEXT,
    week_date TEXT,
    total_feedback INTEGER,
    ces_avg REAL,
    negative_count INTEGER,
    analysis_text TEXT,
    created_at TEXT
);

-- Run log
CREATE TABLE run_log (
    id INTEGER PRIMARY KEY,
    run_date TEXT,
    teams_processed INTEGER,
    teams_failed INTEGER,
    created_at TEXT
);
```

## Cost & Performance

- **Cost**: ~$2-3/month (Claude API)
- **Time saved**: 8-12 hours/month
- **Processing speed**: 1,000 entries in ~30 seconds
- **Accuracy**: 96% severity, 80% category classification

## Limitations

- Requires manual CSV download (can be automated with API)
- Windows Task Scheduler requires logged-in user
- Corporate networks may need SSL bypass
- Classification extraction is heuristic-based

## Future Enhancements (V5)

- Analytics platform API integration (auto-download CSVs)
- GitHub Actions (cloud scheduling)
- Email summaries with trend charts
- Month-over-month rollups
- Real-time Slack notifications

## Contributing

This is a portfolio demonstration project. For production use:

1. Update `teams_config.json` with your teams
2. Customize the prompt template
3. Create your own golden test sets
4. Run evaluations to calibrate accuracy

## License

MIT License - see repository root

## Author

Built by a PM to demonstrate AI product development capabilities.

**Skills demonstrated:**
- AI/ML product development
- Production system design
- Testing & validation frameworks
- Automation & monitoring
- Technical documentation

## Version History

- **V1**: Manual Excel + Claude web interface
- **V2**: Automated API + single team
- **V3**: Multi-team + WoW trending
- **V4**: Validation + evaluation + monitoring
