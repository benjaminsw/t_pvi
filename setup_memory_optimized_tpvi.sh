#!/bin/bash
# setup_memory_optimized_tpvi.sh
# Setup script for memory-optimized T-PVI comparison

echo "🚀 Setting up Memory-Optimized T-PVI Comparison"
echo "================================================"

# Add project to Python path
export PYTHONPATH=$PYTHONPATH:/content

# Create necessary directories
mkdir -p output/sec_5.2
mkdir -p scripts/sec_5.2/config
mkdir -p visualizations

echo "📁 Directories created"

# Check GPU memory
echo "🔧 GPU Memory Status:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Set memory optimization environment variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Use only 70% of GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true       # Allow gradual memory growth
export JAX_ENABLE_X64=false                 # Use 32-bit precision to save memory

echo "💾 Memory optimization settings applied"

# Test basic functionality
echo "🧪 Running memory test..."
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Available devices: {jax.devices()}')
print(f'Device memory: {jax.device_get(jax.devices()[0]).memory_capacity if hasattr(jax.devices()[0], \"memory_capacity\") else \"Unknown\"}')
"

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  1. Test memory optimization:"
echo "     python run_tpvi_comparison_memory_optimized.py test-memory"
echo ""
echo "  2. Run single problem (fastest):"
echo "     python run_tpvi_comparison_memory_optimized.py run-memory-optimized --single-problem banana"
echo ""
echo "  3. Run full comparison:"
echo "     python run_tpvi_comparison_memory_optimized.py run-memory-optimized"
echo ""
echo "💡 Tips to avoid OOM errors:"
echo "  - Start with test-memory command"
echo "  - Use single-problem mode first"
echo "  - Monitor GPU memory with: watch -n 1 nvidia-smi"
echo "  - If still OOM, further reduce n_particles in config"