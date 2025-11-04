# ShieldPrompt Setup Guide

Complete setup instructions for the ShieldPrompt prompt injection defense system.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/shieldprompt.git
cd shieldprompt

# 2. Check system requirements
python shieldprompt/phase0_setup/scripts/check_system.py

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Setup Ollama and download models
python shieldprompt/phase0_setup/scripts/setup_ollama.py

# 5. Test the installation
python shieldprompt/phase0_setup/scripts/test_models.py
```

## Detailed Instructions

### Prerequisites

1. **Python 3.8+**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **Ollama**
   - Visit: https://ollama.ai/
   - Download and install for your OS
   - Verify: `ollama --version`

3. **Git** (for cloning)
   ```bash
   git --version
   ```

4. **Hardware**
   - Minimum: 16GB RAM, 20GB disk space
   - Recommended: 32GB RAM, NVIDIA GPU with 8GB+ VRAM

### Step-by-Step Setup

#### 1. Environment Setup

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Verify System Requirements

Run the system checker:

```bash
python shieldprompt/phase0_setup/scripts/check_system.py
```

This will check:
- Python version
- RAM availability
- GPU presence
- Disk space
- Ollama installation
- Python packages

#### 3. Download LLM Models

Run the setup script:

```bash
python shieldprompt/phase0_setup/scripts/setup_ollama.py
```

This downloads:
- Llama 3.2 (3B) - ~2GB
- Phi-4 (14B) - ~8GB
- Mistral (7B) - ~4GB
- Gemma 2 (9B) - ~5GB
- DeepSeek-R1 (7B) - ~5GB (optional)

**Total: ~25GB**

**Time: 30-60 minutes** (depending on internet speed)

#### 4. Test Models

Run the test script:

```bash
python shieldprompt/phase0_setup/scripts/test_models.py
```

This tests:
- Basic functionality
- Response quality
- Baseline vulnerability
- Performance metrics

**Time: 10-20 minutes**

### Troubleshooting

#### Ollama Not Found

```bash
# Check if Ollama is in PATH
which ollama  # Linux/macOS
where ollama  # Windows

# If not found, install from https://ollama.ai/
```

#### Out of Memory

```bash
# Check available RAM
free -h  # Linux
top  # macOS

# Solutions:
# 1. Close other applications
# 2. Use smaller models first
# 3. Restart your system
```

#### Model Download Fails

```bash
# Try pulling manually
ollama pull llama3.2:3b

# Check Ollama service
ollama list

# Restart Ollama
# Linux: sudo systemctl restart ollama
# macOS/Windows: Restart Ollama app
```

#### Python Package Errors

```bash
# Update pip
pip install --upgrade pip

# Install specific packages
pip install ollama rich transformers torch

# If torch installation fails, try CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# If not working:
# 1. Install/update NVIDIA drivers
# 2. Install CUDA toolkit
# 3. Reinstall PyTorch with CUDA support

pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verification Checklist

After setup, verify:

- [ ] Python 3.8+ installed
- [ ] Ollama installed and accessible
- [ ] All required models downloaded
- [ ] System checker passes
- [ ] Test script completes successfully
- [ ] Results saved to `shieldprompt/phase0_setup/results/`

### What's Next?

Once Phase 0 is complete:

1. **Review Results**: Check `shieldprompt/phase0_setup/results/test_summary.json`
2. **Understand Baselines**: Note which models are more vulnerable
3. **Proceed to Phase 1**: Begin foundation & experimentation

See [Phase-by-Phase Guide](PHASES.md) for next steps.

## Advanced Configuration

### Custom Model Selection

Edit `shieldprompt/phase0_setup/scripts/setup_ollama.py` to customize models:

```python
MODELS = [
    ModelConfig(
        name="your-model",
        tag="version",
        size_gb=X.X,
        description="Description",
        vulnerability_profile="Low/Medium/High",
        required=True
    ),
]
```

### Performance Tuning

For faster inference:

1. **GPU Acceleration**: Ensure CUDA is properly configured
2. **Model Quantization**: Use quantized versions (e.g., `llama3.2:3b-q4_0`)
3. **Batch Processing**: Process multiple prompts in batches

### Docker Setup (Alternative)

Coming in Phase 6 - Full containerized setup with Docker.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/shieldprompt/issues)
- **Documentation**: [docs/](.)
- **Community**: [Discussions](https://github.com/yourusername/shieldprompt/discussions)

---

**Estimated Setup Time**: 2-3 hours total

**Required Downloads**: ~25GB

**Next**: [Phase 1 - Foundation](../shieldprompt/phase1_foundation/README.md)
