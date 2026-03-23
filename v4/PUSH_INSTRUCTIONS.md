# VoCS V4 Portfolio Push Instructions

## Files Downloaded

You should have these files:

**Root V4 folder:**
- `README.md` - V4 documentation
- `PORTFOLIO_README.md` - Main portfolio README (goes in repo root)
- `teams_config.json` - Sanitized team config
- `.env.example` - Environment template
- `team_feedback_prompt_template.txt` - Generic prompt
- `run_vocs.bat.example` - Batch file template

**Tests folder:**
- `tests_eval_runner.py` - Batch evaluation framework
- `tests_golden_test_set.json` - 50 test examples
- `tests_view_trends.py` - Accuracy trend viewer

## What You Still Need to Add

**From your production code, create sanitized versions:**

1. **vocs_multi_team.py** - Your main analyzer
   - Remove: Real API keys, webhook URLs
   - Replace "ATG Entertainment" → "Your Company"
   - Replace product names → generic names
   - Keep: All the logic, structure, code

2. **vocs_validator.py** - Your validation agent
   - Already pretty generic, just copy it

## GitHub Push Steps

### Step 1: Create V4 folder structure

```bash
cd C:\path\to\voc-portfolio-clean

# Create v4 directory
mkdir v4
mkdir v4\tests

# Copy downloaded files
copy path\to\downloads\README.md v4\
copy path\to\downloads\teams_config.json v4\
copy path\to\downloads\.env.example v4\
copy path\to\downloads\team_feedback_prompt_template.txt v4\
copy path\to\downloads\run_vocs.bat.example v4\

# Copy test files
copy path\to\downloads\tests_eval_runner.py v4\tests\eval_runner.py
copy path\to\downloads\tests_golden_test_set.json v4\tests\golden_test_set.json
copy path\to\downloads\tests_view_trends.py v4\tests\view_trends.py
```

### Step 2: Add your production code (sanitized)

**Create `v4\vocs_multi_team.py`:**

```python
# Copy your production vocs_multi_team.py
# Then sanitize it:

# FIND AND REPLACE:
# ATG Entertainment → Your Company
# uk.atgtickets.com → your-company.com
# Audience View → Your Ticketing System
# Adyen → Your Payment Provider

# REMOVE:
# Real Slack webhook URLs (already using env vars, so safe)
# Real Mouseflow URLs (already using config, so safe)
# Any real customer data in comments

# KEEP:
# All the code logic
# All the functions
# All the architecture
```

**Copy `v4\vocs_validator.py`:**
Just copy it - it's already generic!

### Step 3: Update main README

Replace your repo's main README.md with `PORTFOLIO_README.md`:

```bash
copy path\to\downloads\PORTFOLIO_README.md README.md
```

### Step 4: Git commit and push

```bash
cd C:\path\to\voc-portfolio-clean

# Check status
git status

# Add all V4 files
git add v4/
git add README.md

# Commit
git commit -m "Add VoCS V4 - Production excellence with testing & validation

- Batch evaluation framework (96% severity accuracy)
- Golden test sets (50 hand-labeled examples)
- Data validation agent
- Dry-run mode for safe testing
- Historical trend monitoring
- Regression testing
- Fixed all-rows counting bug

Ready for portfolio showcase"

# Push to GitHub
git push origin main
```

## Verification Checklist

Before pushing, verify:

✅ No real API keys in code
✅ No real Slack webhooks (using env vars)
✅ No company-specific product names
✅ No customer data or real feedback
✅ All documentation is professional
✅ Code is well-commented
✅ README explains the project clearly

## Final Structure

```
voc-portfolio-clean/
├── README.md                    (main portfolio overview)
├── v1/                          (existing)
├── v2/                          (existing)
├── v3/                          (existing)
└── v4/                          (NEW)
    ├── README.md
    ├── vocs_multi_team.py       (YOU ADD THIS - sanitized)
    ├── vocs_validator.py        (YOU ADD THIS)
    ├── teams_config.json
    ├── team_feedback_prompt_template.txt
    ├── .env.example
    ├── run_vocs.bat.example
    └── tests/
        ├── eval_runner.py
        ├── golden_test_set.json
        ├── view_trends.py
        └── debug_eval.py        (optional)
```

## Portfolio Talking Points

When showcasing this to VPs/hiring managers:

**What You Built:**
"Production AI system that processes 1,000+ weekly customer surveys across 7 teams with 96% accuracy"

**Impact:**
"Saves 8-12 hours/week, costs $2-3/month, built without engineering resources"

**Skills:**
"AI/ML product development, production system design, automated testing, prompt engineering, evaluation frameworks"

**Evolution:**
"Iterative approach V1→V4 shows product thinking and technical execution"

**Scale:**
"Multi-team platform with config-driven architecture - zero code changes to add teams"

---

## Questions?

If you need help with sanitization or have questions about what to remove, let me know!

Next up: Push to GitHub, then start building that interview prep agent 😎
