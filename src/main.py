"""Driver Drowsiness Detection and Adaptive Braking Simulation - Main Entry Point."""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config, AppConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging across the application."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection and Adaptive Braking Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "video"],
        default=None,
        help="Override video input source.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (if source is video).",
    )
    parser.add_argument(
        "--strategy",
        choices=["abrupt", "gradual", "adaptive"],
        default=None,
        help="Override braking controller strategy.",
    )
    parser.add_argument(
        "--gui",
        choices=["pygame", "opencv", "web"],
        default="web",
        help="Choose GUI interface: web (browser), pygame (desktop), or opencv (native).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for batch benchmarks and headless testing).",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run Monte Carlo batch experiments and generate scientific report and plots.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(args)


def run_batch_experiments(config: AppConfig) -> int:
    """Run Monte Carlo simulation sweeps, detector evaluation, and generate HTML report."""
    from src.analysis.experiments import ExperimentManager
    from src.analysis.evaluate_detector import run_ablation_study
    from src.analysis.plots import (
        plot_speed_vs_time,
        plot_controller_barchart,
        plot_collision_heatmap,
    )
    from src.analysis.report import generate_html_report

    logger = logging.getLogger("experiments")
    logger.info("Executing Monte Carlo parameter sweeps...")
    mgr = ExperimentManager(config)
    report = mgr.run_monte_carlo_sweep(n_samples=30, seed=config.sim.seed)

    logger.info("Generating comparative figures...")
    plot_controller_barchart(report)
    plot_collision_heatmap(report)

    sample_res = {
        "abrupt": report.results[0],
        "gradual": report.results[1],
        "adaptive": report.results[2],
    }
    plot_speed_vs_time(sample_res)

    logger.info("Running detector ablation evaluation...")
    ablation = run_ablation_study()

    logger.info("Synthesizing HTML scientific report...")
    out_path = generate_html_report(report, ablation)
    logger.info("Experiment report successfully generated at: %s", out_path)
    return 0


def main(cli_args: list[str] | None = None) -> int:
    """Main application lifecycle."""
    args = parse_args(cli_args)
    setup_logging(args.verbose)
    logger = logging.getLogger("main")
    logger.info("Initializing Driver Drowsiness Detection & Adaptive Braking Simulation...")

    config: AppConfig = load_config(args.config)

    # Apply CLI overrides if provided
    if args.source:
        config.camera.source = args.source
    if args.video:
        config.camera.video_path = args.video
    if args.strategy:
        config.braking.strategy = args.strategy

    if args.experiments:
        return run_batch_experiments(config)

    if args.headless:
        from src.sim.runner import HeadlessRunner
        logger.info("Running single headless scenario verification...")
        runner = HeadlessRunner(config)
        res = runner.run(strategy=config.braking.strategy)
        logger.info(
            "Headless run complete: Collided=%s, Min Gap=%.2fm, Stopping Dist=%.1fm",
            res.has_collided,
            res.min_gap_m,
            res.stopping_distance_m,
        )
        return 0

    # Interactive Application
    if args.gui == "web":
        from src.ui.web_app import start_web_dashboard
        start_web_dashboard(config)
        return 0
    elif args.gui == "opencv":
        from src.ui.opencv_app import run_opencv_app
        return run_opencv_app(config)

    from src.ui.app import DrowsinessSimulationApp
    app = DrowsinessSimulationApp(config)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
