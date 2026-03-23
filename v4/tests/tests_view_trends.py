"""
View VoCS evaluation accuracy trends over time
"""
import pandas as pd
from pathlib import Path
import sys

def main():
    """Display evaluation trends"""
    results_dir = Path(__file__).parent / "eval_results"
    summary_path = results_dir / "summary.csv"
    
    if not summary_path.exists():
        print("❌ No evaluation results found. Run eval_runner.py first.")
        return 1
    
    df = pd.read_csv(summary_path)
    
    print("\n" + "="*80)
    print("VoCS Classification Accuracy Trends")
    print("="*80)
    print(df.to_string(index=False))
    
    # Check for drift
    if len(df) > 1:
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        cat_drift = latest['category_accuracy'] - previous['category_accuracy']
        sev_drift = latest['severity_accuracy'] - previous['severity_accuracy']
        top3_drift = latest['top3_accuracy'] - previous['top3_accuracy']
        
        print(f"\n" + "="*80)
        print(f"📊 Latest vs Previous:")
        print("="*80)
        print(f"Category accuracy: {cat_drift:+.1f}% change")
        print(f"Severity accuracy: {sev_drift:+.1f}% change")
        print(f"Top-3 accuracy: {top3_drift:+.1f}% change")
        
        # Alert on regression
        if cat_drift < -5:
            print("\n⚠️  REGRESSION DETECTED - category accuracy dropped >5%")
            print(f"   Previous: {previous['category_accuracy']:.1f}%")
            print(f"   Latest: {latest['category_accuracy']:.1f}%")
            return 1
        
        if cat_drift > 5:
            print("\n✅ IMPROVEMENT - category accuracy increased >5%")
    
    else:
        print("\n📊 Only one evaluation run - no trend data yet")
    
    print("\n" + "="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
