"""
System Health Check Utility

Comprehensive health check for MRAG-Bench system to verify all components
are properly configured and operational.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component: str
    status: str  # "pass", "warning", "fail"
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    overall_status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    checks_passed: int
    checks_warned: int
    checks_failed: int
    total_checks: int
    results: List[HealthCheckResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status,
            "timestamp": self.timestamp,
            "checks_passed": self.checks_passed,
            "checks_warned": self.checks_warned,
            "checks_failed": self.checks_failed,
            "total_checks": self.total_checks,
            "results": [asdict(r) for r in self.results]
        }


class SystemHealthCheck:
    """
    Comprehensive system health checker for MRAG-Bench.

    Verifies:
    - Python environment and dependencies
    - GPU and CUDA availability
    - Model accessibility
    - Dataset availability
    - Memory resources
    - Configuration validity
    """

    def __init__(self):
        """Initialize health checker."""
        self.results: List[HealthCheckResult] = []

    def run_all_checks(self) -> SystemHealthReport:
        """
        Run all health checks.

        Returns:
            Complete system health report
        """
        import datetime

        self.results = []

        # Run all checks
        self._check_python_version()
        self._check_dependencies()
        self._check_cuda_availability()
        self._check_gpu_memory()
        self._check_models_accessible()
        self._check_dataset_exists()
        self._check_configuration()
        self._check_directories()

        # Calculate summary
        passed = sum(1 for r in self.results if r.status == "pass")
        warned = sum(1 for r in self.results if r.status == "warning")
        failed = sum(1 for r in self.results if r.status == "fail")
        total = len(self.results)

        # Determine overall status
        if failed > 0:
            overall_status = "unhealthy"
        elif warned > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return SystemHealthReport(
            overall_status=overall_status,
            timestamp=datetime.datetime.now().isoformat(),
            checks_passed=passed,
            checks_warned=warned,
            checks_failed=failed,
            total_checks=total,
            results=self.results
        )

    def _check_python_version(self) -> None:
        """Check Python version compatibility."""
        version = sys.version_info

        if version.major == 3 and version.minor >= 8:
            self.results.append(HealthCheckResult(
                component="Python Version",
                status="pass",
                message=f"Python {version.major}.{version.minor}.{version.micro}",
                details={
                    "version": f"{version.major}.{version.minor}.{version.micro}",
                    "recommended": "3.10+"
                }
            ))
        else:
            self.results.append(HealthCheckResult(
                component="Python Version",
                status="fail",
                message=f"Python {version.major}.{version.minor} not supported (requires 3.8+)",
                details={"version": f"{version.major}.{version.minor}"}
            ))

    def _check_dependencies(self) -> None:
        """Check critical dependencies."""
        critical_deps = [
            "torch",
            "transformers",
            "PIL",
            "numpy",
            "faiss",
            "yaml"
        ]

        missing_deps = []
        imported_deps = {}

        for dep in critical_deps:
            try:
                if dep == "PIL":
                    import PIL
                    imported_deps[dep] = PIL.__version__
                elif dep == "yaml":
                    import yaml
                    imported_deps[dep] = getattr(yaml, '__version__', 'unknown')
                else:
                    module = __import__(dep)
                    imported_deps[dep] = getattr(module, '__version__', 'unknown')
            except ImportError:
                missing_deps.append(dep)

        if not missing_deps:
            self.results.append(HealthCheckResult(
                component="Dependencies",
                status="pass",
                message=f"All {len(critical_deps)} critical dependencies installed",
                details=imported_deps
            ))
        else:
            self.results.append(HealthCheckResult(
                component="Dependencies",
                status="fail",
                message=f"Missing dependencies: {', '.join(missing_deps)}",
                details={"missing": missing_deps, "installed": imported_deps}
            ))

    def _check_cuda_availability(self) -> None:
        """Check CUDA and GPU availability."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()

            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"

                self.results.append(HealthCheckResult(
                    component="CUDA",
                    status="pass",
                    message=f"CUDA {cuda_version} available with {gpu_count} GPU(s)",
                    details={
                        "cuda_version": cuda_version,
                        "gpu_count": gpu_count,
                        "gpu_name": gpu_name,
                        "pytorch_version": torch.__version__
                    }
                ))
            else:
                self.results.append(HealthCheckResult(
                    component="CUDA",
                    status="warning",
                    message="CUDA not available - GPU acceleration disabled",
                    details={"pytorch_version": torch.__version__}
                ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component="CUDA",
                status="fail",
                message=f"Error checking CUDA: {str(e)}",
                details={"error": str(e)}
            ))

    def _check_gpu_memory(self) -> None:
        """Check GPU memory availability."""
        try:
            import torch

            if not torch.cuda.is_available():
                self.results.append(HealthCheckResult(
                    component="GPU Memory",
                    status="warning",
                    message="No GPU available",
                    details={}
                ))
                return

            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / 1e9
            allocated_gb = torch.cuda.memory_allocated(0) / 1e9
            reserved_gb = torch.cuda.memory_reserved(0) / 1e9
            available_gb = total_memory_gb - reserved_gb

            if total_memory_gb >= 16.0:
                status = "pass"
                message = f"GPU memory: {total_memory_gb:.1f}GB (sufficient)"
            elif total_memory_gb >= 12.0:
                status = "warning"
                message = f"GPU memory: {total_memory_gb:.1f}GB (marginal, 16GB recommended)"
            else:
                status = "fail"
                message = f"GPU memory: {total_memory_gb:.1f}GB (insufficient, 16GB required)"

            self.results.append(HealthCheckResult(
                component="GPU Memory",
                status=status,
                message=message,
                details={
                    "total_gb": round(total_memory_gb, 2),
                    "allocated_gb": round(allocated_gb, 2),
                    "reserved_gb": round(reserved_gb, 2),
                    "available_gb": round(available_gb, 2),
                    "gpu_name": gpu_props.name
                }
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component="GPU Memory",
                status="fail",
                message=f"Error checking GPU memory: {str(e)}",
                details={"error": str(e)}
            ))

    def _check_models_accessible(self) -> None:
        """Check if model repositories are accessible."""
        models_to_check = [
            "llava-hf/llava-1.5-7b-hf",
            "openai/clip-vit-base-patch32"
        ]

        accessible_models = []
        inaccessible_models = []

        for model_name in models_to_check:
            try:
                from transformers import AutoConfig
                # Just check if config is accessible (doesn't download full model)
                config = AutoConfig.from_pretrained(model_name)
                accessible_models.append(model_name)
            except Exception as e:
                inaccessible_models.append((model_name, str(e)))

        if not inaccessible_models:
            self.results.append(HealthCheckResult(
                component="Model Access",
                status="pass",
                message=f"All {len(models_to_check)} model repositories accessible",
                details={"accessible": accessible_models}
            ))
        elif len(accessible_models) > 0:
            self.results.append(HealthCheckResult(
                component="Model Access",
                status="warning",
                message=f"Some models inaccessible: {len(inaccessible_models)}/{len(models_to_check)}",
                details={
                    "accessible": accessible_models,
                    "inaccessible": [m[0] for m in inaccessible_models]
                }
            ))
        else:
            self.results.append(HealthCheckResult(
                component="Model Access",
                status="fail",
                message="No models accessible - check internet connection",
                details={"inaccessible": [m[0] for m in inaccessible_models]}
            ))

    def _check_dataset_exists(self) -> None:
        """Check if MRAG-Bench dataset exists."""
        dataset_path = Path("data/mrag_bench")
        images_path = dataset_path / "images"

        if not dataset_path.exists():
            self.results.append(HealthCheckResult(
                component="Dataset",
                status="warning",
                message="Dataset not found - run download_mrag_dataset.py",
                details={"path": str(dataset_path)}
            ))
            return

        # Check for images directory
        if images_path.exists():
            image_count = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.png")))

            if image_count > 15000:
                status = "pass"
                message = f"Dataset found with {image_count} images"
            elif image_count > 0:
                status = "warning"
                message = f"Dataset incomplete: {image_count} images (expected ~16,130)"
            else:
                status = "warning"
                message = "Dataset directory exists but no images found"

            self.results.append(HealthCheckResult(
                component="Dataset",
                status=status,
                message=message,
                details={
                    "path": str(dataset_path),
                    "image_count": image_count
                }
            ))
        else:
            self.results.append(HealthCheckResult(
                component="Dataset",
                status="warning",
                message="Dataset directory exists but images/ subdirectory missing",
                details={"path": str(dataset_path)}
            ))

    def _check_configuration(self) -> None:
        """Check if configuration is valid."""
        config_path = Path("config/mrag_bench.yaml")

        if not config_path.exists():
            self.results.append(HealthCheckResult(
                component="Configuration",
                status="fail",
                message="Configuration file not found",
                details={"path": str(config_path)}
            ))
            return

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.config import MRAGConfig

            config = MRAGConfig.load(str(config_path))
            config.validate()

            self.results.append(HealthCheckResult(
                component="Configuration",
                status="pass",
                message="Configuration loaded and validated successfully",
                details={
                    "path": str(config_path),
                    "vlm": config.model.vlm_name,
                    "retriever": config.model.retriever_name,
                    "quantization": config.model.quantization,
                    "memory_limit": config.performance.memory_limit_gb
                }
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component="Configuration",
                status="fail",
                message=f"Configuration error: {str(e)}",
                details={"path": str(config_path), "error": str(e)}
            ))

    def _check_directories(self) -> None:
        """Check if required directories exist."""
        required_dirs = [
            "data",
            "data/embeddings",
            "output",
            "config"
        ]

        missing_dirs = []
        existing_dirs = []

        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)

        if not missing_dirs:
            self.results.append(HealthCheckResult(
                component="Directories",
                status="pass",
                message=f"All {len(required_dirs)} required directories exist",
                details={"directories": existing_dirs}
            ))
        else:
            self.results.append(HealthCheckResult(
                component="Directories",
                status="warning",
                message=f"Missing directories: {', '.join(missing_dirs)}",
                details={
                    "existing": existing_dirs,
                    "missing": missing_dirs
                }
            ))

    def print_report(self, report: SystemHealthReport) -> None:
        """Print formatted health check report."""
        print("\n" + "=" * 60)
        print("MRAG-Bench System Health Check Report")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Status: {report.overall_status.upper()}")
        print(f"\nSummary:")
        print(f"  ✓ Passed:  {report.checks_passed}/{report.total_checks}")
        print(f"  ! Warnings: {report.checks_warned}/{report.total_checks}")
        print(f"  ✗ Failed:  {report.checks_failed}/{report.total_checks}")
        print("\nDetailed Results:")
        print("-" * 60)

        for result in report.results:
            status_icon = {
                "pass": "✓",
                "warning": "!",
                "fail": "✗"
            }.get(result.status, "?")

            status_color = {
                "pass": "\033[0;32m",  # Green
                "warning": "\033[1;33m",  # Yellow
                "fail": "\033[0;31m"  # Red
            }.get(result.status, "")
            reset_color = "\033[0m"

            print(f"{status_color}{status_icon}{reset_color} {result.component:20s} {result.message}")

        print("\n" + "=" * 60)

        if report.overall_status == "healthy":
            print("✓ System is healthy and ready for operation")
        elif report.overall_status == "degraded":
            print("! System is operational but has warnings")
            print("  Review warnings above and address if possible")
        else:
            print("✗ System has critical issues that must be resolved")
            print("  Review failed checks above and fix before proceeding")

        print("=" * 60 + "\n")


def main():
    """Run health check from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="MRAG-Bench System Health Check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, help="Save report to file")

    args = parser.parse_args()

    checker = SystemHealthCheck()
    report = checker.run_all_checks()

    if args.json:
        output = json.dumps(report.to_dict(), indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)
    else:
        checker.print_report(report)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nJSON report saved to {args.output}")

    # Exit with status code
    # Accept both "healthy" and "degraded" (warnings only) as passing
    sys.exit(0 if report.overall_status in ["healthy", "degraded"] else 1)


if __name__ == "__main__":
    main()
