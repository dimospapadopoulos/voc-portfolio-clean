"""
Voice of Customer Analysis Engine
Processes customer feedback and extracts actionable signals
"""

import anthropic
import pandas as pd
import os
from typing import Dict, List

class FeedbackAnalyzer:
    """Analyzes customer feedback using Claude AI"""
    
    def __init__(self, api_key: str = None):
        """Initialize with Anthropic API key"""
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
    
    def analyze_batch(self, csv_path: str) -> Dict:
        """
        Analyze a batch of feedback entries from CSV
        Returns categorized insights and urgency signals
        """
        # Load feedback data
        df = pd.read_csv(csv_path)
        
        # Prepare high-signal feedback for analysis
        feedback_text = self._prepare_feedback(df)
        
        # Analyze with Claude
        analysis = self._call_claude_api(feedback_text)
        
        # Extract structured signals
        signals = self._extract_signals(analysis, df)
        
        return signals
    
    def _prepare_feedback(self, df: pd.DataFrame) -> str:
        """Format feedback for Claude analysis"""
        # Filter for high-priority signals (score > 7.0)
        high_signal = df[df['signal_score'] > 7.0]
        
        formatted = "High-priority customer feedback:\n\n"
        for _, row in high_signal.iterrows():
            formatted += f"[{row['checkout_stage']}] {row['feedback_text']} (score: {row['signal_score']})\n"
        
        return formatted
    
    def _call_claude_api(self, feedback_text: str) -> str:
        """Send feedback to Claude for analysis"""
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Analyze this customer feedback and identify:
                1. Top 3 themes or patterns
                2. Urgent technical issues requiring attention
                3. UX friction points
                
                Customer Feedback:
                {feedback_text}
                
                Format your response as:
                THEMES: [list themes]
                URGENT ISSUES: [list critical problems]
                UX FRICTION: [list usability problems]"""
            }]
        )
        
        return message.content[0].text
    
    def _extract_signals(self, analysis: str, df: pd.DataFrame) -> Dict:
        """Parse Claude's analysis into structured data"""
        return {
            'summary': analysis,
            'high_priority_count': len(df[df['signal_score'] > 7.0]),
            'total_analyzed': len(df),
            'top_stages': df['checkout_stage'].value_counts().head(3).to_dict(),
            'avg_signal_score': df['signal_score'].mean()
        }

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <path_to_feedback.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print(f"Analyzing feedback from: {csv_path}")
    
    analyzer = FeedbackAnalyzer()
    results = analyzer.analyze_batch(csv_path)
    
    print("\n=== Analysis Results ===")
    print(f"Total entries analyzed: {results['total_analyzed']}")
    print(f"High-priority signals: {results['high_priority_count']}")
    print(f"Average signal score: {results['avg_signal_score']:.2f}")
    print(f"\nTop checkout stages:")
    for stage, count in results['top_stages'].items():
        print(f"  - {stage}: {count}")
    print(f"\n=== Claude Analysis ===")
    print(results['summary'])
    