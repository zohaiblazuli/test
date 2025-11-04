#!/usr/bin/env python3
"""
ShieldPrompt Phase 0: Ollama Model Setup Script

This script downloads and configures the required LLM models for prompt injection testing.
Based on 2025 research, we use models with varying vulnerability profiles.

Models selected:
1. llama3.2:3b - Meta's efficient baseline (3B parameters)
2. phi4:14b - Microsoft's SLM optimized for reasoning (14B parameters)
3. mistral:7b - Excellent open model (7B parameters)
4. gemma2:9b - Google's responsible AI model (9B parameters)
5. deepseek-r1:7b (optional) - Known vulnerable baseline (77% ASR)

Hardware requirements: 16GB+ RAM, 12GB+ VRAM recommended
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

@dataclass
class ModelConfig:
    """Configuration for an Ollama model."""
    name: str
    tag: str
    size_gb: float
    description: str
    vulnerability_profile: str
    required: bool = True

    @property
    def full_name(self) -> str:
        """Return the full model name for Ollama."""
        return f"{self.name}:{self.tag}"


# Model configurations based on 2025 research
MODELS = [
    ModelConfig(
        name="llama3.2",
        tag="3b",
        size_gb=2.0,
        description="Meta Llama 3.2 3B - General baseline model",
        vulnerability_profile="Medium",
        required=True
    ),
    ModelConfig(
        name="phi4",
        tag="14b",
        size_gb=8.0,
        description="Microsoft Phi-4 14B - Efficient SLM for reasoning",
        vulnerability_profile="Low",
        required=True
    ),
    ModelConfig(
        name="mistral",
        tag="7b",
        size_gb=4.1,
        description="Mistral 7B - Balanced performance model",
        vulnerability_profile="Medium-Low",
        required=True
    ),
    ModelConfig(
        name="gemma2",
        tag="9b",
        size_gb=5.4,
        description="Google Gemma 2 9B - Responsible AI model",
        vulnerability_profile="Low",
        required=True
    ),
    ModelConfig(
        name="deepseek-r1",
        tag="7b",
        size_gb=4.7,
        description="DeepSeek-R1 7B - Known vulnerable baseline (77% ASR)",
        vulnerability_profile="High",
        required=False
    ),
]


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_installed_models() -> List[str]:
    """Get list of currently installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output to get model names
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        models = []
        for line in lines:
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except subprocess.CalledProcessError:
        return []


def pull_model(model: ModelConfig, progress: Progress, task_id) -> bool:
    """Pull a model using Ollama."""
    try:
        console.print(f"\n[cyan]Pulling {model.full_name}...[/cyan]")
        console.print(f"[dim]Size: ~{model.size_gb}GB | Profile: {model.vulnerability_profile}[/dim]")

        # Run ollama pull command
        process = subprocess.Popen(
            ["ollama", "pull", model.full_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output
        for line in process.stdout:
            line = line.strip()
            if line:
                # Update progress based on output
                if "pulling" in line.lower():
                    console.print(f"[dim]{line}[/dim]")
                elif "success" in line.lower():
                    console.print(f"[green]✓ {line}[/green]")
                else:
                    console.print(f"[dim]{line}[/dim]")

        process.wait()

        if process.returncode == 0:
            progress.update(task_id, advance=1)
            console.print(f"[green]✓ Successfully pulled {model.full_name}[/green]\n")
            return True
        else:
            console.print(f"[red]✗ Failed to pull {model.full_name}[/red]\n")
            return False

    except Exception as e:
        console.print(f"[red]✗ Error pulling {model.full_name}: {e}[/red]\n")
        return False


def verify_model(model: ModelConfig) -> bool:
    """Verify a model is working by running a simple test."""
    try:
        console.print(f"[cyan]Verifying {model.full_name}...[/cyan]")

        result = subprocess.run(
            ["ollama", "run", model.full_name, "Say 'OK' if you can read this."],
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )

        if result.returncode == 0:
            console.print(f"[green]✓ {model.full_name} verified and working[/green]")
            return True
        else:
            console.print(f"[yellow]⚠ {model.full_name} may have issues[/yellow]")
            return False

    except subprocess.TimeoutExpired:
        console.print(f"[yellow]⚠ {model.full_name} verification timed out[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Error verifying {model.full_name}: {e}[/red]")
        return False


def save_model_config(output_dir: Path):
    """Save model configuration to JSON for later use."""
    config_data = {
        "models": [
            {
                "name": model.name,
                "tag": model.tag,
                "full_name": model.full_name,
                "size_gb": model.size_gb,
                "description": model.description,
                "vulnerability_profile": model.vulnerability_profile,
                "required": model.required,
            }
            for model in MODELS
        ],
        "setup_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": len(MODELS),
        "required_models": len([m for m in MODELS if m.required]),
    }

    config_file = output_dir / "model_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    console.print(f"\n[green]✓ Model configuration saved to {config_file}[/green]")


def display_summary(installed: List[str], successful: List[str], failed: List[str]):
    """Display a summary table of the setup results."""
    table = Table(title="Setup Summary", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", width=20)
    table.add_column("Status", width=15)
    table.add_column("Size", justify="right", width=10)
    table.add_column("Vulnerability", width=15)

    for model in MODELS:
        if model.full_name in successful:
            status = "[green]✓ Installed[/green]"
        elif model.full_name in failed:
            status = "[red]✗ Failed[/red]"
        elif model.full_name in installed:
            status = "[yellow]◆ Pre-existing[/yellow]"
        else:
            status = "[dim]○ Skipped[/dim]"

        table.add_row(
            model.full_name,
            status,
            f"{model.size_gb} GB",
            model.vulnerability_profile
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    """Main setup function."""
    console.print(Panel.fit(
        "[bold cyan]ShieldPrompt Phase 0: Model Setup[/bold cyan]\n"
        "Setting up local LLM infrastructure for prompt injection testing",
        border_style="cyan"
    ))

    # Check if Ollama is installed
    console.print("\n[bold]Step 1: Checking Ollama installation...[/bold]")
    if not check_ollama_installed():
        console.print("[red]✗ Ollama is not installed or not in PATH[/red]")
        console.print("\nPlease install Ollama from: https://ollama.ai/")
        console.print("After installation, run this script again.")
        sys.exit(1)

    console.print("[green]✓ Ollama is installed[/green]")

    # Get currently installed models
    console.print("\n[bold]Step 2: Checking installed models...[/bold]")
    installed_models = get_installed_models()

    if installed_models:
        console.print(f"[green]Found {len(installed_models)} installed model(s)[/green]")
        for model in installed_models:
            console.print(f"  • {model}")
    else:
        console.print("[dim]No models currently installed[/dim]")

    # Determine which models to download
    console.print("\n[bold]Step 3: Planning model downloads...[/bold]")
    to_download = []
    already_installed = []

    for model in MODELS:
        if model.full_name in installed_models:
            already_installed.append(model.full_name)
            console.print(f"[green]✓ {model.full_name} already installed[/green]")
        else:
            to_download.append(model)
            required_text = "Required" if model.required else "Optional"
            console.print(f"[yellow]◆ {model.full_name} - {required_text} ({model.size_gb}GB)[/yellow]")

    if not to_download:
        console.print("\n[green]All required models are already installed![/green]")
        save_model_config(Path("shieldprompt/phase0_setup/data"))
        display_summary(installed_models, [], [])
        return

    # Calculate total download size
    total_size = sum(m.size_gb for m in to_download)
    console.print(f"\n[bold]Total download size: ~{total_size:.1f}GB[/bold]")

    # Ask for confirmation
    console.print("\n[yellow]This will download the models listed above.[/yellow]")
    response = console.input("[bold]Continue? (y/n): [/bold]")

    if response.lower() not in ['y', 'yes']:
        console.print("[yellow]Setup cancelled by user[/yellow]")
        sys.exit(0)

    # Download models
    console.print("\n[bold]Step 4: Downloading models...[/bold]")
    successful = []
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Downloading models...", total=len(to_download))

        for model in to_download:
            if pull_model(model, progress, task):
                successful.append(model.full_name)
            else:
                failed.append(model.full_name)
                if model.required:
                    console.print(f"[red]Warning: Required model {model.full_name} failed to download[/red]")

    # Verify models
    console.print("\n[bold]Step 5: Verifying models...[/bold]")
    for model in to_download:
        if model.full_name in successful:
            verify_model(model)
            time.sleep(1)  # Brief pause between verifications

    # Save configuration
    data_dir = Path("shieldprompt/phase0_setup/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    save_model_config(data_dir)

    # Display summary
    display_summary(already_installed, successful, failed)

    # Final status
    if failed and any(m.full_name in failed for m in MODELS if m.required):
        console.print("[red]⚠ Setup completed with errors. Some required models failed to install.[/red]")
        console.print("[yellow]Please check the errors above and try running the script again.[/yellow]")
        sys.exit(1)
    else:
        console.print(Panel.fit(
            "[bold green]✓ Phase 0 Setup Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Run: python shieldprompt/phase0_setup/scripts/test_models.py\n"
            "2. Review: shieldprompt/phase0_setup/data/model_config.json\n"
            "3. Proceed to Phase 1: Foundation & Experimentation",
            border_style="green"
        ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
