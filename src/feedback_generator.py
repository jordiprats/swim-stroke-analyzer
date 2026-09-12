"""Generates human-readable feedback reports from analysis results."""

from typing import Dict, List
from src.models.freestyle_rules import (
    FreestyleIssue,
    get_severity_emoji as fs_get_severity_emoji,
    get_severity_label as fs_get_severity_label,
    SEVERITY_CRITICAL,
    SEVERITY_MODERATE,
    SEVERITY_MINOR
)
from src.models.butterfly_rules import (
    ButterflyIssue,
    get_severity_emoji as bf_get_severity_emoji,
    get_severity_label as bf_get_severity_label,
)


class FeedbackGenerator:
    """Generates structured feedback reports for swimmers."""

    def generate_report(self, analysis_results: Dict, stroke_type: str = 'freestyle') -> str:
        """
        Generate comprehensive text report.

        Args:
            analysis_results: Results from stroke analyzer
            stroke_type: 'freestyle' or 'butterfly'

        Returns:
            Formatted text report
        """
        metrics = analysis_results['metrics']
        issues = analysis_results['issues']

        is_butterfly = stroke_type == 'butterfly' or 'undulation' in metrics

        report = []

        # Overall rating
        rating = self._calculate_overall_rating(issues)

        # Group issues by severity
        critical_issues = [i for i in issues if i.severity == SEVERITY_CRITICAL]
        moderate_issues = [i for i in issues if i.severity == SEVERITY_MODERATE]
        minor_issues = [i for i in issues if i.severity == SEVERITY_MINOR]

        # Choose severity emoji/label helpers based on stroke type
        if is_butterfly:
            sev_emoji = bf_get_severity_emoji
            sev_label = bf_get_severity_label
        else:
            sev_emoji = fs_get_severity_emoji
            sev_label = fs_get_severity_label

        # ====== HEADER WITH SCORE ======
        stroke_label = "BUTTERFLY" if is_butterfly else "FREESTYLE"
        report.append(f"\U0001f3ca\u200d\u2642\ufe0f YOUR {stroke_label} ANALYSIS")
        report.append("")
        report.append(f"Overall Technique Score: {rating}/10")
        report.append("")

        # ====== QUICK INSIGHT (THE HOOK) ======
        insight = self._generate_quick_insight(rating, critical_issues, moderate_issues, is_butterfly)
        report.append("\u2500" * 36)
        report.append("\U0001f4c9 QUICK INSIGHT")
        report.append("\u2500" * 36)
        report.append(insight)
        report.append("")

        # ====== BIGGEST RED FLAG (if any) ======
        if critical_issues:
            report.append("\U0001f6a8 BIGGEST RED FLAG")
            report.append("\u2500" * 36)
            top_issue = critical_issues[0]
            report.append(f"\u26a0\ufe0f  {top_issue.description}")
            report.append("")
            report.append("\U0001f4a1 HOW TO FIX IT:")
            report.append(f"   {top_issue.tip}")
            report.append("")
            if len(critical_issues) > 1:
                report.append(f"   (+ {len(critical_issues) - 1} more critical issue{'s' if len(critical_issues) > 2 else ''} detected)")
            report.append("")

        # ====== WHAT'S WORKING ======
        strengths = self._identify_strengths(metrics, issues)
        if strengths:
            report.append("\u2705 WHAT'S WORKING")
            report.append("\u2500" * 36)
            for strength in strengths:
                report.append(f"\u2022 {strength}")
            report.append("")

        # ====== ACTION PLAN ======
        if critical_issues or moderate_issues:
            report.append("\U0001f3af YOUR ACTION PLAN")
            report.append("\u2500" * 36)
            report.append("Focus on these in order:")
            report.append("")

            # Add critical issues
            for i, issue in enumerate(critical_issues, 1):
                report.append(f"{i}. \U0001f6a8 FIX THIS FIRST: {issue.description}")
                report.append(f"   \u2192 {issue.tip}")
                report.append("")

            # Add top moderate issues
            start_num = len(critical_issues) + 1
            for i, issue in enumerate(moderate_issues[:2], start_num):  # Only top 2 moderate
                report.append(f"{i}. \u26a0\ufe0f IMPORTANT: {issue.description}")
                report.append(f"   \u2192 {issue.tip}")
                report.append("")

        # ====== DETAILED BREAKDOWN (Collapsed by default in UI) ======
        if critical_issues or moderate_issues or minor_issues:
            report.append("\U0001f4cb DETAILED BREAKDOWN")
            report.append("\u2500" * 36)

            # Critical issues
            if critical_issues:
                report.append(f"{sev_emoji(SEVERITY_CRITICAL)} {sev_label(SEVERITY_CRITICAL)}:")
                for issue in critical_issues:
                    report.append(f"  \u2022 {issue.description}")
                    report.append(f"    \u2192 {issue.tip}")
                report.append("")

            # Moderate issues
            if moderate_issues:
                report.append(f"{sev_emoji(SEVERITY_MODERATE)} {sev_label(SEVERITY_MODERATE)}:")
                for issue in moderate_issues:
                    report.append(f"  \u2022 {issue.description}")
                    report.append(f"    \u2192 {issue.tip}")
                report.append("")

            # Minor issues
            if minor_issues:
                report.append(f"{sev_emoji(SEVERITY_MINOR)} {sev_label(SEVERITY_MINOR)}:")
                for issue in minor_issues:
                    report.append(f"  \u2022 {issue.description}")
                    report.append(f"    \u2192 {issue.tip}")
                report.append("")

        # ====== METRICS ======
        report.append("\U0001f4ca YOUR NUMBERS")
        report.append("\u2500" * 36)
        report.append(self._format_metrics(metrics))
        report.append("")

        # ====== NO ISSUES CELEBRATION ======
        if not issues:
            stroke_noun = "butterfly" if is_butterfly else "freestyle"
            report.append("\U0001f3c6 EXCELLENT TECHNIQUE!")
            report.append("\u2500" * 36)
            report.append(f"No major technique issues detected! Your {stroke_noun} form is solid.")
            report.append("Keep up the great work and maintain that consistency!")
            report.append("")

        # ====== FOOTER WITH TIP ======
        report.append("\u2500" * 36)
        report.append("\U0001f4a1 PRO TIP: Focus on fixing one issue at a time.")
        report.append("   Trying to change everything at once = slower progress!")
        report.append("\u2500" * 36)

        return "\n".join(report)

    def generate_summary(self, analysis_results: Dict, stroke_type: str = 'freestyle') -> str:
        """
        Generate brief summary.

        Args:
            analysis_results: Results from stroke analyzer
            stroke_type: 'freestyle' or 'butterfly'

        Returns:
            Brief text summary
        """
        issues = analysis_results['issues']
        rating = self._calculate_overall_rating(issues)

        critical_count = len([i for i in issues if i.severity == SEVERITY_CRITICAL])
        moderate_count = len([i for i in issues if i.severity == SEVERITY_MODERATE])

        stroke_label = stroke_type.upper()
        summary = [
            f"{stroke_label} Score: {rating}/10",
            f"Critical Issues: {critical_count}",
            f"Areas for Improvement: {moderate_count}"
        ]

        return " | ".join(summary)

    def _calculate_overall_rating(self, issues: List) -> int:
        """
        Calculate overall technique rating (1-10).

        Args:
            issues: List of detected issues (FreestyleIssue or ButterflyIssue)

        Returns:
            Rating from 1-10
        """
        # Start with perfect score
        score = 10

        # Deduct points based on severity
        for issue in issues:
            if issue.severity == SEVERITY_CRITICAL:
                score -= 2
            elif issue.severity == SEVERITY_MODERATE:
                score -= 1
            elif issue.severity == SEVERITY_MINOR:
                score -= 0.5

        return max(1, min(10, int(round(score))))

    def _generate_quick_insight(self, rating: int, critical_issues: List, moderate_issues: List, is_butterfly: bool = False) -> str:
        """Generate a quick, shareable insight about the swim."""
        if rating >= 9:
            return "\U0001f3c6 Your technique is Olympic-level! Maintain this form and focus on consistency."
        elif rating >= 7:
            if critical_issues:
                return f"\U0001f4aa Solid foundation, but you're losing speed/efficiency due to: {critical_issues[0].issue_type.replace('_', ' ')}. Fix that and you'll see big gains!"
            else:
                return "\U0001f44d Good technique overall! A few tweaks and you'll be swimming like a pro."
        elif rating >= 5:
            if critical_issues:
                return f"\u26a0\ufe0f  Your biggest issue is: {critical_issues[0].issue_type.replace('_', ' ')}. This is costing you the most energy and speed. Focus here first!"
            else:
                return "\U0001f527 Several areas need work, but they're all fixable! Follow the action plan below."
        else:
            if critical_issues:
                return f"\U0001f6a8 Red flag alert: {critical_issues[0].issue_type.replace('_', ' ')}. This is significantly impacting your swim. Let's fix it step by step!"
            else:
                return "\U0001f4da You're just getting started! Follow the action plan and you'll see improvement quickly."

    def _identify_strengths(self, metrics: Dict, issues: List) -> List[str]:
        """Identify what the swimmer is doing well."""
        strengths = []

        # Detect stroke type from available metrics
        is_butterfly = 'undulation' in metrics and metrics['undulation'].get('undulation_amplitude') is not None

        if is_butterfly:
            # --- Butterfly strengths ---
            if metrics.get('elbow', {}).get('avg_angle'):
                angle = metrics['elbow']['avg_angle']
                if 120 <= angle <= 150:
                    strengths.append("Good elbow bend during pull — maintaining leverage!")

            if metrics.get('undulation', {}).get('undulation_amplitude'):
                amp = metrics['undulation']['undulation_amplitude']
                if 0.04 <= amp <= 0.10:
                    strengths.append("Nice dolphin undulation — efficient wave motion!")

            if metrics.get('synchronization', {}).get('synchronized') is True:
                strengths.append("Arms are well synchronised — both hands entering together!")

            if metrics.get('head', {}).get('breathing_lift'):
                lift = metrics['head']['breathing_lift']
                if 0.05 <= lift <= 0.08:
                    strengths.append("Controlled breathing — minimal head lift!")

        else:
            # --- Freestyle strengths ---
            if metrics.get('elbow', {}).get('avg_angle'):
                angle = metrics['elbow']['avg_angle']
                if 80 <= angle <= 100:
                    strengths.append("Great elbow catch angle — you're engaging your lats properly!")

            if metrics.get('rotation', {}).get('avg_rotation'):
                rotation = metrics['rotation']['avg_rotation']
                if 45 <= rotation <= 60:
                    strengths.append("Excellent body rotation — you're using your core effectively!")

            if metrics.get('head', {}).get('stability'):
                stability = metrics['head']['stability']
                if stability > 0.8:
                    strengths.append("Solid head position — you're maintaining good alignment!")

            if metrics.get('stroke_rate', {}).get('spm'):
                spm = metrics['stroke_rate']['spm']
                if 50 <= spm <= 60:
                    strengths.append("Optimal stroke rate — nice rhythm and tempo!")

        # If no specific strengths but also few issues
        if not strengths and len(issues) <= 2:
            strengths.append("Your overall form is consistent across the video!")

        return strengths

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics section (handles both freestyle and butterfly)."""
        lines = []

        # Detect stroke type
        is_butterfly = 'undulation' in metrics and metrics['undulation'].get('undulation_amplitude') is not None

        # Elbow metrics
        if metrics.get('elbow', {}).get('avg_angle') is not None:
            elbow = metrics['elbow']
            if is_butterfly:
                lines.append(f"\U0001f7b8 Elbow Pull Angle:")
                lines.append(f"   Average: {elbow['avg_angle']:.1f}\u00b0 (optimal: 120-160\u00b0)")
            else:
                lines.append(f"\U0001f7b8 Elbow Catch Angle:")
                lines.append(f"   Average: {elbow['avg_angle']:.1f}\u00b0 (optimal: 80-100\u00b0)")
            if elbow['left_avg'] and elbow['right_avg']:
                lines.append(f"   Left: {elbow['left_avg']:.1f}\u00b0 | Right: {elbow['right_avg']:.1f}\u00b0")
            lines.append("")

        if is_butterfly:
            # Butterfly-specific metrics
            if metrics.get('entry', {}).get('avg_entry_width') is not None:
                entry = metrics['entry']
                lines.append(f"\U0001f7b8 Arm Entry Width:")
                lines.append(f"   Avg wrist span: {entry['avg_entry_width']:.2f}x shoulder width")
                lines.append("")

            if metrics.get('undulation', {}).get('undulation_amplitude') is not None:
                und = metrics['undulation']
                lines.append(f"\U0001f7b8 Dolphin Undulation:")
                lines.append(f"   Amplitude: {und['undulation_amplitude']:.3f} (optimal: 0.04-0.12)")
                lines.append(f"   Cycles detected: {und['undulation_cycles']}")
                lines.append("")

            if metrics.get('synchronization', {}).get('sync_delta') is not None:
                sync = metrics['synchronization']
                status = "Synchronised" if sync['synchronized'] else "Not synchronised"
                lines.append(f"\U0001f7b8 Arm Synchronisation:")
                lines.append(f"   Avg delay: {sync['sync_delta']:.2f}s ({status})")
                lines.append("")
        else:
            # Freestyle-specific metrics
            if metrics.get('rotation', {}).get('avg_rotation') is not None:
                rotation = metrics['rotation']
                lines.append(f"\U0001f7b8 Body Rotation:")
                lines.append(f"   Average: {rotation['avg_rotation']:.1f}\u00b0 (optimal: 45-60\u00b0)")
                lines.append(f"   Range: {rotation['min_rotation']:.1f}\u00b0 - {rotation['max_rotation']:.1f}\u00b0")
                lines.append("")

        # Stroke rate (shared by both)
        if metrics.get('stroke_rate', {}).get('spm') is not None:
            sr = metrics['stroke_rate']
            if is_butterfly:
                lines.append(f"\U0001f7b8 Stroke Rate:")
                lines.append(f"   {sr['spm']:.1f} strokes/min (optimal butterfly: 30-55 SPM)")
            else:
                lines.append(f"\U0001f7b8 Stroke Rate:")
                lines.append(f"   {sr['spm']:.1f} strokes/min (optimal: 50-60 SPM)")
            lines.append(f"   Duration: {sr['duration']:.1f}s | Total strokes: {sr['total_strokes']}")
            lines.append("")

        # Head stability / breathing
        if metrics.get('head', {}).get('stability') is not None:
            head = metrics['head']
            stability_pct = head['stability'] * 100
            lines.append(f"\U0001f7b8 Head Stability: {stability_pct:.1f}% (higher is better)")
            lines.append("")

        if metrics.get('head', {}).get('breathing_lift') is not None:
            head = metrics['head']
            lines.append(f"\U0001f7b8 Breathing Lift: {head['breathing_lift']:.2f} of frame height (optimal < 0.08)")
            lines.append("")

        # Kick
        if metrics.get('kick', {}).get('avg_knee_angle') is not None:
            kick = metrics['kick']
            if is_butterfly:
                lines.append(f"\U0001f7b8 Dolphin Kick:")
                lines.append(f"   Knee angle: {kick['avg_knee_angle']:.1f}\u00b0 (optimal: 120-150\u00b0)")
            else:
                lines.append(f"\U0001f7b8 Kick Mechanics:")
                lines.append(f"   Knee angle: {kick['avg_knee_angle']:.1f}\u00b0 (should be near 170\u00b0)")
            lines.append("")

        # Video quality
        if metrics.get('valid_frame_ratio') is not None:
            valid_pct = metrics['valid_frame_ratio'] * 100
            lines.append(f"\U0001f7b8 Detection Quality: {valid_pct:.1f}% of frames analyzed")

        return "\n".join(lines)
