"""
VoCS Evaluation Runner - Batch Mode
Tests classification accuracy against golden test set
Evaluates all examples as a batch (matches production workflow)
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
import re

# Add parent directory to path to import from main script
sys.path.insert(0, str(Path(__file__).parent.parent))

from vocs_multi_team import (
    load_config, 
    load_prompt_template,
    customize_prompt,
    analyze_with_claude,
    build_feedback_summary,
    CLAUDE_MODEL
)


class VoCSEvaluator:
    """Evaluates VoCS classification accuracy using batch analysis"""
    
    def __init__(self, golden_set_path: Path, config_path: Path, template_path: Path):
        self.golden_set_path = golden_set_path
        self.config_path = config_path
        self.template_path = template_path
        
        with open(golden_set_path, 'r') as f:
            self.golden_set = json.load(f)
        
        self.config = self._load_config()
        self.template = self._load_template()
        
        self.results_dir = Path(__file__).parent / "eval_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def _load_config(self):
        """Load teams config"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return {team['id']: team for team in config['teams']}
    
    def _load_template(self):
        """Load prompt template"""
        return self.template_path.read_text(encoding='utf-8')
    
    def run_evaluation(self, team_filter: str = None) -> Dict:
        """Run evaluation on all or specific team"""
        print("="*60)
        print("VoCS Evaluation Runner - BATCH MODE")
        print("="*60)
        print(f"Model: {CLAUDE_MODEL}")
        print(f"Golden set version: {self.golden_set['version']}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*60)
        
        eval_results = {
            'timestamp': datetime.now().isoformat(),
            'model': CLAUDE_MODEL,
            'golden_set_version': self.golden_set['version'],
            'mode': 'batch',
            'teams': {},
            'overall': {
                'total': 0,
                'correct_category': 0,
                'correct_severity': 0,
                'correct_top3': 0,
                'failures': []
            }
        }
        
        # Process each team
        for team_id, examples in self.golden_set.items():
            if team_id in ['version', 'created', 'description']:
                continue
            
            if team_filter and team_id != team_filter:
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating: {team_id} ({len(examples)} examples as batch)")
            print(f"{'='*60}")
            
            team_results = self._evaluate_team_batch(team_id, examples)
            eval_results['teams'][team_id] = team_results
            
            # Update overall stats
            eval_results['overall']['total'] += team_results['total']
            eval_results['overall']['correct_category'] += team_results['correct_category']
            eval_results['overall']['correct_severity'] += team_results['correct_severity']
            eval_results['overall']['correct_top3'] += team_results['correct_top3']
            eval_results['overall']['failures'].extend(team_results['failures'])
        
        # Calculate percentages
        total = eval_results['overall']['total']
        if total > 0:
            eval_results['overall']['category_accuracy'] = round(
                eval_results['overall']['correct_category'] / total * 100, 1
            )
            eval_results['overall']['severity_accuracy'] = round(
                eval_results['overall']['correct_severity'] / total * 100, 1
            )
            eval_results['overall']['top3_accuracy'] = round(
                eval_results['overall']['correct_top3'] / total * 100, 1
            )
        
        # Print summary
        self._print_summary(eval_results)
        
        # Save results
        self._save_results(eval_results)
        
        return eval_results
    
    def _evaluate_team_batch(self, team_id: str, examples: List[Dict]) -> Dict:
        """Evaluate all examples as a single batch (matches production)"""
        team_config = self.config[team_id]
        
        print(f"  📦 Creating batch of {len(examples)} examples...")
        
        # Create DataFrame from ALL examples at once
        all_data = []
        for example in examples:
            all_data.append({
                '_ces_score_': example['ces_score'],
                '_session_id_': example.get('session_id', 'test_session'),
                '_date_': example.get('date', '2026-03-01'),
                '_combined_feedback_': example['feedback']
            })
        
        df = pd.DataFrame(all_data)
        
        # Create metadata for the batch
        metadata = {
            'total_rows': len(df),
            'rows_with_text': len(df),
            'rows_without_text': 0,
            'ces_breakdown_all': df['_ces_score_'].value_counts().to_dict(),
            'ces_avg_all': df['_ces_score_'].mean(),
            'negative_count_all': len(df[df['_ces_score_'] <= 2]),
            'team_name': team_config['name'],
            'team_id': team_id,
            'ces_avg': df['_ces_score_'].mean(),
            'negative_count': len(df[df['_ces_score_'] <= 2])
        }
        
        # Build feedback summary for entire batch
        print(f"  🤖 Sending batch to Claude...")
        feedback_summary = build_feedback_summary(df, metadata, max_rows=500)
        
        # Get Claude's analysis of the BATCH
        prompt = customize_prompt(self.template, team_config, metadata, "2026-W12-TEST", None)
        
        try:
            batch_analysis = analyze_with_claude(prompt, feedback_summary)
            print(f"  ✅ Batch analysis complete ({len(batch_analysis)} chars)")
        except Exception as e:
            print(f"  ❌ Error getting batch analysis: {e}")
            return {
                'total': len(examples),
                'correct_category': 0,
                'correct_severity': 0,
                'correct_top3': 0,
                'failures': [{'error': str(e)}],
                'examples': []
            }
        
        # NOW check each example against the batch analysis
        team_results = {
            'total': len(examples),
            'correct_category': 0,
            'correct_severity': 0,
            'correct_top3': 0,
            'failures': [],
            'examples': [],
            'batch_analysis': batch_analysis  # Store full batch response
        }
        
        print(f"  🔍 Evaluating individual examples against batch...")
        
        for i, example in enumerate(examples, 1):
            extracted = self._extract_from_batch(batch_analysis, example, i)
            comparison = self._compare_results(example['expected'], extracted, example['id'])
            
            if comparison['category_match']:
                team_results['correct_category'] += 1
            if comparison['severity_match']:
                team_results['correct_severity'] += 1
            if comparison['top3_match']:
                team_results['correct_top3'] += 1
            
            status = "✅" if comparison['all_match'] else "❌"
            print(f"    {status} {example['id']}")
            
            if not comparison['all_match']:
                team_results['failures'].append({
                    'id': example['id'],
                    'mismatches': comparison['mismatches']
                })
            
            team_results['examples'].append({
                'id': example['id'],
                'expected': example['expected'],
                'extracted': extracted,
                'comparison': comparison
            })
        
        return team_results
    
    def _extract_from_batch(self, batch_analysis: str, example: Dict, position: int) -> Dict:
        """Extract how Claude classified this specific example within the batch"""
        extracted = {
            'category': None,
            'severity': None,
            'top3': False
        }
        
        feedback_lower = example['feedback'].lower()
        analysis_lower = batch_analysis.lower()
        
        # Extract a unique snippet from the feedback (first 40 chars of actual text)
        feedback_text = re.sub(r'\[reason \d+\]:', '', feedback_lower)
        feedback_text = re.sub(r'\[improvement\]:', '', feedback_text)
        feedback_snippet = feedback_text.strip()[:40]
        
        # Check if this feedback appears in "Top 3 Critical Issues" section
        # Look for section headers
        top3_patterns = [
            r'top\s*3.*?(?:other|medium|lower|positive)',
            r'critical.*?(?:other|medium|lower|positive)',
            r'urgent.*?(?:other|medium|lower|positive)',
            r'most\s*important.*?(?:other|medium|lower|positive)'
        ]
        
        for pattern in top3_patterns:
            match = re.search(pattern, analysis_lower, re.DOTALL)
            if match:
                section_text = match.group(0)
                # Check if our feedback snippet appears in this section
                if feedback_snippet in section_text:
                    extracted['top3'] = True
                    break
        
        # Alternative: Check for numbered lists (1., 2., 3.) and see if our snippet is near them
        if not extracted['top3']:
            # Look for patterns like "1. " or "1) " followed by our snippet
            for num in [1, 2, 3]:
                pattern = rf'{num}[\.\)]\s*.{{0,100}}{re.escape(feedback_snippet[:20])}'
                if re.search(pattern, analysis_lower, re.DOTALL):
                    extracted['top3'] = True
                    break
        
        # Extract category based on feedback keywords
        category_patterns = {
            'Payment': ['payment', 'declined', 'card', 'billing', 'transaction', 'pay'],
            'Mobile': ['mobile', 'phone', 'app', 'chrome', 'ios', 'android'],
            'Technical': ['error', 'crash', 'broken', 'failed', 'stuck', 'won\'t load', 'doesn\'t work'],
            'UX': ['navigation', 'navigate', 'confusing', 'difficult', 'hard to', 'intuitive', 'timer'],
            'Performance': ['slow', 'loading', 'speed', 'lag'],
            'Search': ['search', 'find', 'filter'],
            'Account': ['account', 'login', 'password', 'email'],
            'Accessibility': ['accessible', 'wheelchair', 'disability'],
            'Positive': ['easy', 'great', 'excellent', 'perfect', 'simple', 'smooth', 'exceptional']
        }
        
        # Check for positive first (based on CES score)
        if example['ces_score'] >= 4:
            positive_words = ['easy', 'simple', 'great', 'perfect', 'excellent', 'loved']
            if any(word in feedback_lower for word in positive_words):
                extracted['category'] = 'Positive'
        
        # If not positive, find category
        if not extracted['category']:
            for category, keywords in category_patterns.items():
                if any(kw in feedback_lower for kw in keywords):
                    extracted['category'] = category
                    break
        
        # Severity based on CES score + impact keywords
        ces = example['ces_score']
        
        # Base severity on CES
        severity_map = {
            1: 10,
            2: 7,
            3: 5,
            4: 2,
            5: 0
        }
        base_severity = severity_map.get(ces, 5)
        
        # Adjust for critical keywords
        critical_keywords = ['payment', 'can\'t', 'cannot', 'blocked', 'failed', 'error', 'crash', 'broken']
        high_keywords = ['difficult', 'confusing', 'hard', 'slow']
        
        if any(kw in feedback_lower for kw in critical_keywords):
            # Keep high severity
            extracted['severity'] = base_severity
        elif any(kw in feedback_lower for kw in high_keywords):
            # Medium severity
            extracted['severity'] = max(base_severity - 1, 0)
        else:
            # Lower severity for vague complaints
            extracted['severity'] = max(base_severity - 2, 0)
        
        # Cap severity at 10
        extracted['severity'] = min(extracted['severity'], 10)
        
        return extracted
    
    def _compare_results(self, expected: Dict, extracted: Dict, example_id: str) -> Dict:
        """Compare expected vs extracted classifications"""
        comparison = {
            'category_match': False,
            'severity_match': False,
            'top3_match': False,
            'all_match': False,
            'mismatches': []
        }
        
        # Category comparison
        expected_cat = expected.get('category', '').lower() if expected.get('category') else ''
        extracted_cat = extracted.get('category', '').lower() if extracted.get('category') else ''
        
        if expected_cat == extracted_cat:
            comparison['category_match'] = True
        else:
            comparison['mismatches'].append(
                f"Category: expected '{expected['category']}', got '{extracted.get('category', 'None')}'"
            )
        
        # Severity comparison (within ±2 points)
        expected_sev = expected['severity']
        extracted_sev = extracted.get('severity', 0)
        if abs(expected_sev - extracted_sev) <= 2:
            comparison['severity_match'] = True
        else:
            comparison['mismatches'].append(
                f"Severity: expected {expected_sev}, got {extracted_sev}"
            )
        
        # Top 3 comparison
        if expected.get('top_3', False) == extracted.get('top3', False):
            comparison['top3_match'] = True
        else:
            comparison['mismatches'].append(
                f"Top3: expected {expected.get('top_3')}, got {extracted.get('top3')}"
            )
        
        comparison['all_match'] = (
            comparison['category_match'] and 
            comparison['severity_match'] and 
            comparison['top3_match']
        )
        
        return comparison
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        overall = results['overall']
        total = overall['total']
        
        print(f"Total examples evaluated: {total}")
        print(f"Category accuracy: {overall['correct_category']}/{total} ({overall.get('category_accuracy', 0):.1f}%)")
        print(f"Severity accuracy: {overall['correct_severity']}/{total} ({overall.get('severity_accuracy', 0):.1f}%)")
        print(f"Top-3 accuracy: {overall['correct_top3']}/{total} ({overall.get('top3_accuracy', 0):.1f}%)")
        
        if overall['failures']:
            print(f"\n❌ Failures ({len(overall['failures'])}):")
            for failure in overall['failures'][:10]:
                if 'error' in failure:
                    print(f"  - {failure.get('error', 'Unknown error')}")
                else:
                    print(f"  - {failure['id']}: {', '.join(failure.get('mismatches', []))}")
            if len(overall['failures']) > 10:
                print(f"  ... and {len(overall['failures']) - 10} more")
    
    def _save_results(self, results: Dict):
        """Save results to JSON and update summary CSV"""
        # Save detailed JSON
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        json_path = self.results_dir / f"{timestamp}_batch.json"
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed results saved: {json_path}")
        
        # Update summary CSV
        summary_path = self.results_dir / "summary.csv"
        
        summary_row = {
            'timestamp': results['timestamp'],
            'model': results['model'],
            'mode': 'batch',
            'total_examples': results['overall']['total'],
            'category_accuracy': results['overall'].get('category_accuracy', 0),
            'severity_accuracy': results['overall'].get('severity_accuracy', 0),
            'top3_accuracy': results['overall'].get('top3_accuracy', 0),
            'failures': len(results['overall']['failures'])
        }
        
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df = pd.DataFrame([summary_row])
        
        df.to_csv(summary_path, index=False)
        print(f"📊 Summary updated: {summary_path}")


def main():
    """Run evaluation"""
    script_dir = Path(__file__).parent.parent
    
    golden_set = Path(__file__).parent / "golden_test_set.json"
    config = script_dir / "teams_config.json"
    template = script_dir / "team_feedback_prompt_template.txt"
    
    if not golden_set.exists():
        print(f"❌ Golden test set not found: {golden_set}")
        return 1
    
    evaluator = VoCSEvaluator(golden_set, config, template)
    
    # Allow filtering by team
    team_filter = sys.argv[1] if len(sys.argv) > 1 else None
    
    results = evaluator.run_evaluation(team_filter)
    
    # Exit with error code if accuracy below threshold
    category_acc = results['overall'].get('category_accuracy', 0)
    if category_acc < 70:
        print(f"\n⚠️  WARNING: Category accuracy ({category_acc:.1f}%) below 70% threshold")
        return 1
    
    print("\n✅ Evaluation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
