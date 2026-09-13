#!/usr/bin/env python3
"""
Swim Stroke Analyzer - Main CLI Entry Point

Analyzes freestyle or butterfly swimming technique from video
and provides coaching feedback.
"""

import argparse
import sys
import os
from pathlib import Path

from src.pose_detector import PoseDetector
from src.video_processor import VideoProcessor
from src.stroke_analyzer import StrokeAnalyzer
from src.butterfly_analyzer import ButterflyAnalyzer
from src.visualizer import Visualizer
from src.feedback_generator import FeedbackGenerator
from src.data_exporter import export_datapoints_csv


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze swimming technique from video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a video
  python main.py video.mp4

  # Analyze butterfly
  python main.py video.mp4 --stroke butterfly

  # Generate report only
  python main.py video.mp4 --report-only
        """
    )

    parser.add_argument(
        'video',
        help='Path to input video file'
    )

    parser.add_argument(
        '--stroke',
        choices=['freestyle', 'butterfly'],
        default='freestyle',
        help='Type of stroke to analyze (default: freestyle)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Path for output video (default: input_analyzed.mp4)',
        default=None
    )

    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate text report only, skip video visualization'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip text report, only generate annotated video'
    )


    parser.add_argument(
        '--analyze-every',
        type=int,
        default=1,
        help='Process every Nth frame (1 = every frame, 2 = every other). Default: 1'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Set output path
    if args.output is None:
        args.output = VideoProcessor.generate_output_path(args.video)

    print("=" * 60)
    print(f"SWIM STROKE ANALYZER — {args.stroke.upper()}")
    print("=" * 60)
    print(f"Input video: {args.video}")
    if not args.report_only:
        print(f"Output video: {args.output}")
    print("")

    try:
        # Step 1: Extract pose data from video
        print("Step 1/4: Detecting pose in video frames...")
        detector = PoseDetector()
        pose_data = detector.process_video(args.video, skip_frames=args.analyze_every)

        if not pose_data:
            print("Error: Failed to process video")
            sys.exit(1)

        print(f"✓ Processed {len(pose_data)} frames")
        print("")

        # ── Always export datapoints CSV for offline debugging ──
        datapoints_path = os.path.splitext(args.output)[0] + '_datapoints.csv'
        export_datapoints_csv(pose_data, datapoints_path)
        print(f"✓ Datapoints saved to {datapoints_path}")
        print("")


        # Step 2: Analyze stroke mechanics
        stroke_label = args.stroke.capitalize()
        print(f"Step 2/4: Analyzing {stroke_label} stroke mechanics...")

        if args.stroke == 'butterfly':
            analyzer = ButterflyAnalyzer()
        else:
            analyzer = StrokeAnalyzer()

        analysis_results = analyzer.analyze_video(pose_data)

        if 'error' in analysis_results:
            print(f"Error: {analysis_results['error']}")
            sys.exit(1)

        print("✓ Analysis complete")
        print("")

        # Step 3: Generate feedback report
        if not args.no_report:
            print("Step 3/4: Generating feedback report...")
            feedback = FeedbackGenerator()
            report = feedback.generate_report(analysis_results)
            print("✓ Report generated")
            print("")
            print(report)
            print("")

            # Save report to file
            report_path = os.path.splitext(args.output)[0] + '_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {report_path}")
            print("")
        else:
            print("Step 3/4: Skipping report generation")
            print("")

        # Step 4: Create annotated video
        if not args.report_only:
            print("Step 4/4: Creating annotated video...")
            visualizer = Visualizer()
            output_video = visualizer.create_annotated_video(
                pose_data,
                args.output,
                analysis_results,
                args.video
            )
            print(f"✓ Video saved to: {output_video}")

            # Best-effort H.264 re-encode for browser playback
            if VideoProcessor.reencode_for_browser(output_video):
                print("✓ Re-encoded to H.264 for browser compatibility")
            else:
                print("  (ffmpeg not found — skipping browser re-encode)")
            print("")
        else:
            print("Step 4/4: Skipping video generation")
            print("")

        print("=" * 60)
        print(f"{args.stroke.upper()} ANALYSIS COMPLETE")
        print("=" * 60)

        # Quick summary
        feedback = FeedbackGenerator()
        summary = feedback.generate_summary(analysis_results, args.stroke)
        print(summary)
        print("")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
