# Phase 0: Setup Local LLM Infrastructure

**Week 1 - November 2025**

## Overview

Phase 0 establishes the foundation for the entire ShieldPrompt project by setting up a local LLM testing environment. This phase ensures you have all necessary models downloaded, configured, and tested before beginning vulnerability research.

## Goals

- ✅ Install and configure Ollama
- ✅ Download 4-5 open-source LLM models with varying vulnerability profiles
- ✅ Verify all models work correctly
- ✅ Establish baseline performance metrics
- ✅ Document the setup for reproducibility

## Hardware Requirements

### Minimum
- 16GB RAM
- Decent CPU (most modern laptops work)
- 20GB free disk space

### Recommended (Your Setup)
- 32GB RAM ✓
- NVIDIA RTX 3060 12GB VRAM ✓
- 50GB free disk space

Your hardware is excellent for this project!

## Models Selected

Based on 2025 research into prompt injection vulnerabilities:

| Model | Size | Purpose | Vulnerability Profile |
|-------|------|---------|---------------------|
| **Llama 3.2** | 3B | General baseline | Medium |
| **Phi-4** | 14B | Efficient SLM reasoning | Low |
| **Mistral** | 7B | Balanced performance | Medium-Low |
| **Gemma 2** | 9B | Responsible AI testing | Low |
| **DeepSeek-R1** | 7B | Known vulnerable baseline | High (77% ASR) |

### Why These Models?

1. **Llama 3.2** - Industry standard, widely used, good documentation
2. **Phi-4** - Microsoft's efficient model, tests SLM vulnerabilities
3. **Mistral** - Excellent open model, balanced test case
4. **Gemma 2** - Google's safety-focused model, tests responsible AI claims
5. **DeepSeek-R1** - Known to have high vulnerability rate, useful as a "control" case

## Setup Instructions

### Step 1: Verify Ollama Installation

```bash
# Check if Ollama is installed
ollama --version

# If not installed, visit: https://ollama.ai/
```

### Step 2: Run Setup Script

```bash
# Navigate to project root
cd /home/user/test

# Run the setup script
python shieldprompt/phase0_setup/scripts/setup_ollama.py
```

This script will:
- Check for existing models
- Download required models (~25GB total)
- Verify each model works
- Save configuration to `data/model_config.json`

**Time estimate:** 30-60 minutes (depending on internet speed)

### Step 3: Test All Models

```bash
# Run comprehensive tests
python shieldprompt/phase0_setup/scripts/test_models.py
```

This script tests:
- Basic functionality (math, reasoning, creativity)
- Response quality
- Baseline vulnerability to simple injections
- Performance metrics (latency, tokens/sec)

**Time estimate:** 10-20 minutes

## Output Files

After successful setup, you'll have:

```
shieldprompt/phase0_setup/
├── data/
│   └── model_config.json          # Model configurations
├── results/
│   ├── test_results.json          # Detailed test results
│   └── test_summary.json          # Summary statistics
└── scripts/
    ├── setup_ollama.py            # Setup script
    └── test_models.py             # Testing script
```

## Expected Results

### Model Configuration (`data/model_config.json`)

Contains metadata for each model:
- Model name and tag
- Size in GB
- Vulnerability profile
- Setup date

### Test Results (`results/test_results.json`)

For each model and test:
- Prompt and response
- Response time
- Token generation speed
- Injection success/failure

### Test Summary (`results/test_summary.json`)

Aggregate statistics:
- Vulnerability rates per model
- Average response times
- Performance metrics

## Verification Checklist

- [ ] Ollama is installed and accessible
- [ ] All 4-5 models downloaded successfully
- [ ] Each model responds to basic prompts
- [ ] Test script completes without errors
- [ ] `model_config.json` exists
- [ ] `test_results.json` and `test_summary.json` created
- [ ] You understand the baseline vulnerability rates

## Troubleshooting

### Model Download Fails

```bash
# Try pulling manually
ollama pull llama3.2:3b

# Check Ollama service status
ollama list
```

### Out of Memory Errors

- Close other applications
- Try smaller models first (llama3.2:3b, gemma2:2b)
- Restart Ollama service

### Slow Performance

- Ensure GPU is being used (check with `nvidia-smi`)
- Reduce concurrent model loads
- Check available VRAM

### Python Dependencies Missing

```bash
# Install required packages
pip install -r requirements.txt

# Install specific package
pip install ollama rich
```

## Key Insights from Phase 0

After completing Phase 0, you should understand:

1. **Model Variability**: Different models have different vulnerability profiles
2. **Performance Baselines**: How fast each model responds
3. **Simple Injections**: Which basic attacks work on which models
4. **Setup Process**: How to configure reproducible LLM environments

These insights are crucial for Phase 1-2 where you'll build upon this foundation.

## Next Steps

Once Phase 0 is complete:

1. **Review Results**: Analyze vulnerability rates in `test_summary.json`
2. **Document Findings**: Note which models are most/least vulnerable
3. **Begin Phase 1**: Move to foundation & experimentation
   - Literature review on prompt injection
   - Create attack database
   - Build testing framework

## Resources

- [Ollama Documentation](https://ollama.ai/)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [Phi-4 Technical Report](https://huggingface.co/microsoft/phi-4)
- [Mistral Documentation](https://docs.mistral.ai/)
- [Gemma 2 Model Card](https://huggingface.co/google/gemma-2-9b)
- [OWASP LLM Top 10](https://genai.owasp.org/llmrisk/)

## Time Investment

- **Setup**: 1-2 hours
- **Testing**: 30 minutes
- **Analysis**: 30 minutes
- **Total**: ~2-3 hours

## Deliverables

- ✅ Working local LLM setup
- ✅ Test script that queries all models
- ✅ Setup documentation (this file)
- ✅ Baseline vulnerability data

---

**Phase 0 Status:** Ready to Execute ✅

**Next Phase:** Phase 1 - Foundation & Experimentation (Weeks 2-3)
