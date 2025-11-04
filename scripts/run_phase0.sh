#!/bin/bash
# ShieldPrompt Phase 0 - Complete Setup Script
# This script runs all Phase 0 components in sequence

set -e  # Exit on error

echo "========================================="
echo "ShieldPrompt Phase 0: Complete Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${CYAN}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Step 1: Check system requirements
print_step "Step 1: Checking system requirements..."
if python shieldprompt/phase0_setup/scripts/check_system.py; then
    print_success "System check passed"
else
    print_error "System check failed"
    print_warning "Please review the output above and fix any issues"
    exit 1
fi

echo ""
echo "Press Enter to continue to model setup, or Ctrl+C to abort..."
read

# Step 2: Setup Ollama models
print_step "Step 2: Setting up Ollama models..."
echo "This will download ~25GB of models. It may take 30-60 minutes."
echo ""

if python shieldprompt/phase0_setup/scripts/setup_ollama.py; then
    print_success "Model setup completed"
else
    print_error "Model setup failed"
    exit 1
fi

echo ""
echo "Press Enter to continue to testing, or Ctrl+C to abort..."
read

# Step 3: Test all models
print_step "Step 3: Testing all models..."
echo "This will run comprehensive tests. It may take 10-20 minutes."
echo ""

if python shieldprompt/phase0_setup/scripts/test_models.py; then
    print_success "Model testing completed"
else
    print_error "Model testing failed"
    exit 1
fi

# Step 4: Summary
echo ""
echo "========================================="
echo -e "${GREEN}Phase 0 Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  • shieldprompt/phase0_setup/data/model_config.json"
echo "  • shieldprompt/phase0_setup/results/test_results.json"
echo "  • shieldprompt/phase0_setup/results/test_summary.json"
echo ""
echo "Next steps:"
echo "  1. Review the test results"
echo "  2. Read the Phase 1 documentation"
echo "  3. Begin foundation & experimentation"
echo ""
print_success "You're ready to proceed to Phase 1!"
