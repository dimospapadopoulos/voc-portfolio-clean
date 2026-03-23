"""
VoCS Validation Agent
Checks data integrity and raises alerts for anomalies
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime


class VoCSValidator:
    """Validates VoCS data processing pipeline"""
    
    def __init__(self, teams_config_path: Path):
        with open(teams_config_path, 'r') as f:
            self.config = json.load(f)
        self.teams = self.config['teams']
        self.validation_results = []
    
    def validate_all_teams(self, input_dir: Path) -> Dict:
        """Run validation checks for all teams"""
        print("="*60)
        print("VoCS Validation Agent")
        print("="*60)
        
        all_passed = True
        summary = {
            'timestamp': datetime.now().isoformat(),
            'teams_validated': 0,
            'teams_passed': 0,
            'teams_failed': 0,
            'warnings': [],
            'errors': []
        }
        
        for team in self.teams:
            team_result = self.validate_team(team, input_dir)
            summary['teams_validated'] += 1
            
            if team_result['passed']:
                summary['teams_passed'] += 1
                print(f"  ✅ {team['name']}: PASSED")
            else:
                summary['teams_failed'] += 1
                all_passed = False
                print(f"  ❌ {team['name']}: FAILED")
                for error in team_result['errors']:
                    print(f"     - {error}")
                    summary['errors'].append(f"{team['name']}: {error}")
            
            if team_result['warnings']:
                for warning in team_result['warnings']:
                    print(f"  ⚠️  {team['name']}: {warning}")
                    summary['warnings'].append(f"{team['name']}: {warning}")
        
        print("\n" + "="*60)
        print(f"VALIDATION SUMMARY")
        print("="*60)
        print(f"Teams validated: {summary['teams_validated']}")
        print(f"✅ Passed: {summary['teams_passed']}")
        print(f"❌ Failed: {summary['teams_failed']}")
        print(f"⚠️  Warnings: {len(summary['warnings'])}")
        
        return summary
    
    def validate_team(self, team_config: Dict, input_dir: Path) -> Dict:
        """Validate a single team's data"""
        result = {
            'team_id': team_config['id'],
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Find CSV file
        pattern = team_config['csv_pattern']
        csv_files = list(input_dir.glob(pattern))
        
        if not csv_files:
            result['passed'] = False
            result['errors'].append(f"No CSV found matching pattern: {pattern}")
            return result
        
        if len(csv_files) > 1:
            result['warnings'].append(f"Multiple CSVs found, using most recent")
        
        csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
        
        try:
            df = pd.read_csv(csv_path)
            result['metrics']['csv_path'] = str(csv_path)
            result['metrics']['total_rows'] = len(df)
            
            # Check 1: Total row count
            if len(df) == 0:
                result['passed'] = False
                result['errors'].append("CSV file is empty")
                return result
            
            # Check 2: Required columns exist
            col_map = team_config.get('columns', {})
            missing_cols = []
            
            for logical_name, actual_name in col_map.items():
                if actual_name not in df.columns:
                    missing_cols.append(f"{logical_name} ({actual_name})")
            
            if missing_cols:
                result['passed'] = False
                result['errors'].append(f"Missing columns: {', '.join(missing_cols)}")
                return result
            
            # Check 3: Count feedback entries
            feedback_cols = [
                col_map.get('feedback_1'),
                col_map.get('feedback_2'),
                col_map.get('feedback_3')
            ]
            feedback_cols = [c for c in feedback_cols if c and c in df.columns]
            
            # Count rows with ANY feedback
            has_feedback = df[feedback_cols].fillna('').astype(str).apply(
                lambda row: any(str(val).strip() != '' for val in row), axis=1
            )
            rows_with_feedback = has_feedback.sum()
            
            result['metrics']['rows_with_feedback'] = int(rows_with_feedback)
            result['metrics']['rows_without_feedback'] = int(len(df) - rows_with_feedback)
            result['metrics']['feedback_rate'] = round(rows_with_feedback / len(df) * 100, 1)
            
            # CRITICAL CHECK: Total count validation
            result['metrics']['total_rows_in_csv'] = len(df)
            
            # Check that we're not accidentally filtering too early
            print(f"    CSV total: {len(df)} rows")
            print(f"    With feedback: {rows_with_feedback} rows")
            print(f"    Scores only: {len(df) - rows_with_feedback} rows")
            
            if len(df) - rows_with_feedback > len(df) * 0.7:
                result['warnings'].append(
                    f"Low comment rate: {(rows_with_feedback/len(df)*100):.1f}% "
                    f"({rows_with_feedback}/{len(df)})"
                )
            
            # Check 4: Validate CES scores
            ces_col = col_map.get('ces_score')
            if ces_col in df.columns:
                ces_values = df[ces_col].dropna()
                invalid_ces = ces_values[(ces_values < 1) | (ces_values > 5)]
                
                if len(invalid_ces) > 0:
                    result['warnings'].append(f"{len(invalid_ces)} rows have invalid CES scores (not 1-5)")
                
                result['metrics']['ces_breakdown'] = df[ces_col].value_counts().to_dict()
                result['metrics']['avg_ces'] = round(df[ces_col].mean(), 2)
            
            # Check 5: Session ID validation
            session_col = col_map.get('session_id')
            if session_col in df.columns:
                null_sessions = df[session_col].isna().sum()
                if null_sessions > 0:
                    result['warnings'].append(f"{null_sessions} rows missing session IDs")
                
                duplicate_sessions = df[session_col].duplicated().sum()
                if duplicate_sessions > 0:
                    result['warnings'].append(f"{duplicate_sessions} duplicate session IDs")
            
            # Check 6: Anomaly detection - sudden volume changes
            if rows_with_feedback < 10:
                result['warnings'].append(f"Unusually low feedback volume: {rows_with_feedback} entries")
            
            # Check 7: Date validation
            date_col = col_map.get('date')
            if date_col in df.columns:
                try:
                    dates = pd.to_datetime(df[date_col], errors='coerce')
                    invalid_dates = dates.isna().sum()
                    if invalid_dates > 0:
                        result['warnings'].append(f"{invalid_dates} rows have invalid dates")
                except Exception as e:
                    result['warnings'].append(f"Could not parse dates: {e}")
            
        except Exception as e:
            result['passed'] = False
            result['errors'].append(f"Error reading CSV: {str(e)}")
        
        return result
    
    def save_validation_report(self, summary: Dict, output_path: Path):
        """Save validation results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Validation report saved to: {output_path}")


def main():
    """Run validation"""
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    config_path = script_dir / "teams_config.json"
    input_dir = script_dir / "input"
    
    validator = VoCSValidator(config_path)
    summary = validator.validate_all_teams(input_dir)
    
    # Save report
    report_path = script_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    validator.save_validation_report(summary, report_path)
    
    # Exit with error code if validation failed
    if summary['teams_failed'] > 0:
        print("\n❌ Validation FAILED - do not proceed with analysis")
        return 1
    else:
        print("\n✅ Validation PASSED - safe to run analysis")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())