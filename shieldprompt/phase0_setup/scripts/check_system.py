#!/usr/bin/env python3
"""
System Requirements Checker for ShieldPrompt

Checks if your system meets the requirements for running the project.
"""

import subprocess
import platform
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_ram_size() -> str:
    """Get total RAM size."""
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024 ** 2)
                        return f"{ram_gb:.1f} GB"
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True,
                text=True
            )
            ram_bytes = int(result.stdout.strip())
            ram_gb = ram_bytes / (1024 ** 3)
            return f"{ram_gb:.1f} GB"
        return "Unknown"
    except:
        return "Unknown"


def get_gpu_info() -> str:
    """Get GPU information."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "No NVIDIA GPU detected"
    except FileNotFoundError:
        return "nvidia-smi not found (no NVIDIA GPU or drivers not installed)"


def get_disk_space(path: str = ".") -> str:
    """Get available disk space."""
    try:
        if platform.system() == "Linux" or platform.system() == "Darwin":
            result = subprocess.run(
                ['df', '-h', path],
                capture_output=True,
                text=True
            )
            lines = result.stdout.split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                return parts[3]  # Available space
        return "Unknown"
    except:
        return "Unknown"


def check_python_version() -> tuple[str, bool]:
    """Check Python version."""
    version = sys.version.split()[0]
    major, minor = map(int, version.split('.')[:2])
    meets_req = (major == 3 and minor >= 8)
    return version, meets_req


def check_ollama() -> tuple[str, bool]:
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return version, True
        return "Not found", False
    except FileNotFoundError:
        return "Not installed", False


def check_python_packages() -> dict:
    """Check if required Python packages are installed."""
    packages = {
        'ollama': False,
        'torch': False,
        'transformers': False,
        'rich': False,
    }

    for package in packages.keys():
        try:
            __import__(package)
            packages[package] = True
        except ImportError:
            packages[package] = False

    return packages


def main():
    """Main system check function."""
    console.print(Panel.fit(
        "[bold cyan]ShieldPrompt System Requirements Check[/bold cyan]\n"
        "Verifying your system is ready for the project",
        border_style="cyan"
    ))

    # System Information Table
    sys_table = Table(title="System Information", show_header=True, header_style="bold magenta")
    sys_table.add_column("Component", style="cyan", width=25)
    sys_table.add_column("Details", width=40)
    sys_table.add_column("Status", width=15)

    # Operating System
    os_name = f"{platform.system()} {platform.release()}"
    sys_table.add_row("Operating System", os_name, "[green]✓[/green]")

    # Python Version
    py_version, py_ok = check_python_version()
    status = "[green]✓ OK[/green]" if py_ok else "[red]✗ Need 3.8+[/red]"
    sys_table.add_row("Python Version", py_version, status)

    # RAM
    ram = get_ram_size()
    ram_gb = float(ram.split()[0]) if ram != "Unknown" else 0
    ram_status = "[green]✓ OK[/green]" if ram_gb >= 16 else "[yellow]⚠ Low (need 16GB+)[/yellow]"
    sys_table.add_row("RAM", ram, ram_status)

    # GPU
    gpu = get_gpu_info()
    gpu_status = "[green]✓ Available[/green]" if "NVIDIA" in gpu else "[yellow]○ Optional[/yellow]"
    sys_table.add_row("GPU", gpu, gpu_status)

    # Disk Space
    disk = get_disk_space()
    sys_table.add_row("Available Disk Space", disk, "[cyan]ℹ Need 20GB+[/cyan]")

    # Ollama
    ollama_ver, ollama_ok = check_ollama()
    ollama_status = "[green]✓ Installed[/green]" if ollama_ok else "[red]✗ Not found[/red]"
    sys_table.add_row("Ollama", ollama_ver, ollama_status)

    console.print("\n")
    console.print(sys_table)

    # Python Packages Table
    pkg_table = Table(title="Python Dependencies", show_header=True, header_style="bold magenta")
    pkg_table.add_column("Package", style="cyan", width=25)
    pkg_table.add_column("Status", width=20)

    packages = check_python_packages()
    for pkg, installed in packages.items():
        status = "[green]✓ Installed[/green]" if installed else "[yellow]○ Not installed[/yellow]"
        pkg_table.add_row(pkg, status)

    console.print("\n")
    console.print(pkg_table)

    # Overall Assessment
    console.print("\n")

    issues = []
    warnings = []

    if not py_ok:
        issues.append("Python 3.8+ is required")
    if not ollama_ok:
        issues.append("Ollama is not installed")
    if ram_gb < 16 and ram_gb > 0:
        warnings.append("Less than 16GB RAM may cause issues with larger models")
    if not any(packages.values()):
        issues.append("No required Python packages installed (run: pip install -r requirements.txt)")

    if issues:
        console.print(Panel(
            "[bold red]Issues Found:[/bold red]\n\n" +
            "\n".join(f"• {issue}" for issue in issues),
            border_style="red",
            title="❌ Action Required"
        ))
    elif warnings:
        console.print(Panel(
            "[bold yellow]Warnings:[/bold yellow]\n\n" +
            "\n".join(f"• {warning}" for warning in warnings) +
            "\n\nYou can proceed, but monitor resource usage.",
            border_style="yellow",
            title="⚠ Proceed with Caution"
        ))
    else:
        console.print(Panel(
            "[bold green]✓ All requirements met![/bold green]\n\n"
            "Your system is ready for ShieldPrompt.\n\n"
            "Next steps:\n"
            "1. Run: python shieldprompt/phase0_setup/scripts/setup_ollama.py\n"
            "2. Run: python shieldprompt/phase0_setup/scripts/test_models.py",
            border_style="green",
            title="✅ System Ready"
        ))

    # Hardware Recommendations
    console.print("\n[bold]Recommended Configuration:[/bold]")
    console.print("  Minimum: 16GB RAM, 20GB disk, CPU only")
    console.print("  Optimal: 32GB RAM, 50GB disk, NVIDIA GPU 8GB+ VRAM")
    console.print(f"\n[bold]Your system: {ram} RAM, GPU: {gpu.split(',')[0] if ',' in gpu else 'None'}[/bold]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[red]Error during system check: {e}[/red]")
        import traceback
        traceback.print_exc()
