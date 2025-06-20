# run_tpvi_comparison_memory_optimized.py
# Memory-optimized version to avoid CUDA OOM errors

import os
import gc
import typer
from tqdm import tqdm
from src.problems.toy import *
from src.id import *
from src.trainers.trainer import trainer
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from pathlib import Path
from src.base import *
from src.utils import (make_step_and_carry,
                       config_to_parameters,
                       parse_config)
import pickle
from ot.sliced import sliced_wasserstein_distance
import numpy
from mmdfuse import mmdfuse
import pandas as pd
import jax
import jax.numpy as np


app = typer.Typer()

PROBLEMS = {
    'banana': Banana,
    'multimodal': Multimodal,
    'xshape': XShape,
}

ALGORITHMS = ['tpvi', 'pvi', 'svi', 'uvi', 'sm']  # All algorithms

# Memory optimization settings
MEMORY_OPTIMIZED_CONFIG = {
    'n_particles': 50,          # Reduced from 100
    'n_hidden': 256,            # Reduced from 512
    'n_updates': 5000,          # Reduced from 15000
    'n_reruns': 3,              # Reduced from 10
    'mc_n_samples': 100,        # Reduced from 250
    'batch_size': 1,            # Process one problem at a time
    'clear_cache_freq': 100,    # Clear JAX cache every N iterations
}

def clear_jax_cache():
    """Clear JAX compilation cache to free GPU memory"""
    try:
        import jax._src.compilation_cache as cc
        cc.clear_cache()
    except:
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Clear JAX device arrays
    try:
        jax.clear_backends()
    except:
        pass

def memory_safe_metrics_fn(key, target, id, n_samples=500, n_retries=50):
    """Memory-efficient version of metrics computation"""
    # Reduce sample sizes to save memory
    avg_rej = 0
    for _ in range(n_retries):
        m_key, t_key, test_key, key = jax.random.split(key, 4)
        model_samples = id.sample(m_key, n_samples, None)
        target_samples = target.sample(t_key, n_samples, None)
        avg_rej = avg_rej + mmdfuse(test_key, model_samples, target_samples)
    
    # Compute Wasserstein distance with fewer samples
    distance = 0
    n_w_retries = 3  # Reduced retries
    w_samples = 5000  # Reduced samples
    for _ in range(n_w_retries):
        m_key, t_key, key = jax.random.split(key, 3)
        model_samples = id.sample(m_key, w_samples, None)
        target_samples = target.sample(t_key, w_samples, None)
        distance = distance + sliced_wasserstein_distance(
            numpy.array(model_samples), numpy.array(target_samples),
            n_projections=50,  # Reduced projections
        )
    
    return {'power': avg_rej / n_retries,
            'sliced_w': distance / n_w_retries}

def create_memory_optimized_config(base_config):
    """Create a memory-optimized version of the config"""
    optimized_config = base_config.copy()
    
    # Apply memory optimizations to both algorithms
    for algo in ['pvi', 'tpvi']:
        if algo in optimized_config:
            # Reduce model size
            optimized_config[algo]['model']['n_particles'] = MEMORY_OPTIMIZED_CONFIG['n_particles']
            optimized_config[algo]['model']['n_hidden'] = MEMORY_OPTIMIZED_CONFIG['n_hidden']
            
            # Reduce Monte Carlo samples
            if 'extra_alg' in optimized_config[algo]:
                if 'mc_n_samples' in optimized_config[algo]['extra_alg']:
                    optimized_config[algo]['extra_alg']['mc_n_samples'] = MEMORY_OPTIMIZED_CONFIG['mc_n_samples']
    
    # Reduce experiment parameters
    optimized_config['experiment']['n_updates'] = MEMORY_OPTIMIZED_CONFIG['n_updates']
    optimized_config['experiment']['n_reruns'] = MEMORY_OPTIMIZED_CONFIG['n_reruns']
    
    return optimized_config

def memory_safe_visualize(key, ids, target, path, prefix=""):
    """Memory-safe visualization with reduced resolution"""
    _max = 4.5
    _min = -4.5
    x_lin = np.linspace(_min, _max, 200)  # Reduced from 1000
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)
    log_p = lambda x: target.log_prob(x, None)
    log_true_ZZ = vmap(vmap(log_p))(XY)
    
    plt.clf()
    
    # Special handling for PVI and T-PVI comparison
    if 'pvi' in ids and 'tpvi' in ids:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Reduced figure size
        
        # True distribution
        c_true = axes[0].contour(
            XX, YY, np.exp(log_true_ZZ),
            levels=5, colors='black', linewidths=2)
        axes[0].set_title('True Distribution')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # PVI
        model_log_p = lambda x: ids['pvi'].log_prob(x, None)
        log_model_ZZ = vmap(vmap(model_log_p))(XY)
        axes[1].contour(XX, YY, np.exp(log_true_ZZ), levels=5, colors='black', linewidths=1, alpha=0.5)
        axes[1].contour(XX, YY, np.exp(log_model_ZZ), levels=5, colors='blue', linewidths=2)
        axes[1].set_title('PVI')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # T-PVI
        model_log_p = lambda x: ids['tpvi'].log_prob(x, None)
        log_model_ZZ = vmap(vmap(model_log_p))(XY)
        axes[2].contour(XX, YY, np.exp(log_true_ZZ), levels=5, colors='black', linewidths=1, alpha=0.5)
        axes[2].contour(XX, YY, np.exp(log_model_ZZ), levels=5, colors='red', linewidths=2)
        axes[2].set_title('T-PVI')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(path / f"{prefix}_pvi_tpvi_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del XX, YY, XY, log_true_ZZ, log_model_ZZ
        clear_jax_cache()

@app.command()
def run_memory_optimized(config_name: str = "toy-tpvi-comparison",
                        seed: int = 2,
                        single_problem: str = None,
                        algorithms: str = "tpvi,pvi"):
    """Run memory-optimized T-PVI comparison
    
    Args:
        config_name: Configuration file name
        seed: Random seed
        single_problem: Run only one problem (banana, multimodal, xshape)
        algorithms: Comma-separated list of algorithms (tpvi,pvi,svi,uvi,sm)
    """
    
    try:
        print("🚀 Starting Memory-Optimized T-PVI Comparison")
        print("=" * 60)
        
        # Parse algorithms list
        algorithms_to_run = [algo.strip() for algo in algorithms.split(',')]
        all_algorithms = ['tpvi', 'pvi', 'svi', 'uvi', 'sm']
        
        # Validate algorithms
        invalid_algos = [algo for algo in algorithms_to_run if algo not in all_algorithms]
        if invalid_algos:
            print(f"❌ Invalid algorithms: {invalid_algos}")
            print(f"   Valid options: {all_algorithms}")
            return
        
        print(f"🎯 Running algorithms: {algorithms_to_run}")
        
        # Load and optimize config
        config_path = Path(f"scripts/sec_5.2/config/{config_name}.yaml")
        
        # Check if config exists, if not try to use the memory-optimized one from documents
        if not config_path.exists():
            print(f"⚠️  Config file {config_path} not found")
            print("    Creating default memory-optimized config...")
            
            # Create the config directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use the memory-optimized config from the documents
            base_config = {
                'experiment': {
                    'n_reruns': 3,
                    'n_updates': 3000,
                    'name': 'tpvi_memory_optimized',
                    'compute_metrics': False,
                    'use_jit': True
                },
                'pvi': {
                    'algorithm': 'pvi',
                    'model': {
                        'use_particles': True,
                        'n_particles': 50,
                        'd_z': 2,
                        'kernel': 'norm_fixed_var_w_skip',
                        'n_hidden': 256
                    },
                    'theta_opt': {
                        'lr': 1e-4,
                        'optimizer': 'rmsprop',
                        'lr_decay': False,
                        'regularization': 1e-8,
                        'clip': False
                    },
                    'r_opt': {
                        'lr': 1e-2,
                        'regularization': 1e-8
                    },
                    'extra_alg': {
                        'fudge': 0,
                        'mc_n_samples': 100
                    }
                },
                'tpvi': {
                    'algorithm': 'tpvi',
                    'model': {
                        'use_particles': True,
                        'n_particles': 50,
                        'd_z': 2,
                        'kernel': 'norm_fixed_var_w_skip',
                        'n_hidden': 256
                    },
                    'theta_opt': {
                        'lr': 1e-4,
                        'optimizer': 'rmsprop',
                        'lr_decay': False,
                        'regularization': 1e-8,
                        'clip': False
                    },
                    'r_opt': {
                        'lr': 1e-2,
                        'regularization': 1e-8
                    },
                    'extra_alg': {
                        'fudge': 0,
                        'mc_n_samples': 100,
                        'annealing_schedule': "polynomial",
                        'lambda_0': 1e-2,
                        'lambda_min': 1e-8,
                        'gamma': 0.55,
                        'alpha': 0.01,
                        't0': 500,
                        'tau': 50,
                        'monitor_entropy': True,
                        'entropy_threshold': 0.1,
                        'diversity_threshold': 0.01
                    }
                }
            }
            print("    ✅ Using built-in memory-optimized configuration")
        else:
            base_config = parse_config(config_path)
            print(f"    ✅ Loaded config from {config_path}")
        
        config = create_memory_optimized_config(base_config)
        
        print("Memory Optimizations Applied:")
        print(f"  - Particles: {MEMORY_OPTIMIZED_CONFIG['n_particles']}")
        print(f"  - Hidden units: {MEMORY_OPTIMIZED_CONFIG['n_hidden']}")
        print(f"  - Updates: {MEMORY_OPTIMIZED_CONFIG['n_updates']}")
        print(f"  - Reruns: {MEMORY_OPTIMIZED_CONFIG['n_reruns']}")
        print(f"  - MC samples: {MEMORY_OPTIMIZED_CONFIG['mc_n_samples']}")
        
        print("\n🔧 Setting up experiment parameters...")
        
        # Continue with the main execution
        n_rerun = config['experiment']['n_reruns']
        n_updates = config['experiment']['n_updates']
        name = config['experiment']['name']
        name = 'memory_optimized' if len(name) == 0 else f"{name}_memory_opt"
        use_jit = config['experiment']['use_jit']
        print(f"    ✅ Experiment params: {n_rerun} reruns, {n_updates} updates, JIT={use_jit}")

        print("🗂️  Setting up output directory...")
        parent_path = Path(f"output/sec_5.2/{name}")
        parent_path.mkdir(parents=True, exist_ok=True)
        print(f"    ✅ Output directory: {parent_path}")
        
        print("🎲 Initializing random key...")
        
        # Force CPU mode immediately to avoid GPU hanging
        print("    🖥️  Forcing CPU-only mode to avoid GPU issues...")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
        print("    🔧 Configuring JAX for CPU...")
        # Configure JAX outside try-except to avoid scoping issues
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update('jax_enable_x64', False)  # Use 32-bit for faster CPU
        print("    ✅ JAX configured for CPU")
        
        print("    🎲 Creating PRNG key (CPU mode)...")
        key = jax.random.PRNGKey(seed)
        print(f"    ✅ Random seed: {seed} (CPU mode)")
        
        # Test basic JAX functionality
        print("    🔍 Testing basic JAX operations...")
        test_key1, test_key2 = jax.random.split(key, 2)
        print("    ✅ Key splitting works")
        
        test_array = jax.random.normal(test_key1, (2, 2))
        print(f"    ✅ Random generation works: shape {test_array.shape}")
        
        print("📊 Setting up data structures...")
        histories = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        print("    ✅ Data structures initialized")
        
        print("🧪 Setting up problems to run...")
        # Filter problems if specified
        problems_to_run = PROBLEMS
        if single_problem and single_problem in PROBLEMS:
            problems_to_run = {single_problem: PROBLEMS[single_problem]}
            print(f"    ✅ Single problem mode: {single_problem}")
        else:
            print(f"    ✅ All problems mode: {list(problems_to_run.keys())}")

        print(f"\n🔍 Final setup summary:")
        print(f"    Problems: {list(problems_to_run.keys())}")
        print(f"    Algorithms: {algorithms_to_run}")
        print(f"    Training: {n_rerun} runs × {n_updates} updates")
        print(f"    Output: {parent_path}")
        
        print("\n🚀 Starting main training loop...")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return

    # Main training loop with better error handling
    try:
        print("🔄 Entering main training loop...")
        for prob_idx, (prob_name, problem) in enumerate(problems_to_run.items()):
            print(f"\n{'='*20} {prob_name.upper()} ({'='*20}")
            print(f"📝 Problem {prob_idx + 1}/{len(problems_to_run)}: {prob_name}")
            
            try:
                print(f"    🎯 Creating target problem instance...")
                target_test = problem()
                print(f"    ✅ Target created: dim={target_test.dim}, de={target_test.de}")
                del target_test  # Clean up test instance
            except Exception as e:
                print(f"    ❌ Failed to create target: {e}")
                continue
            try:
                print(f"    🎯 Creating target problem instance...")
                target_test = problem()
                print(f"    ✅ Target created: dim={target_test.dim}, de={target_test.de}")
                del target_test  # Clean up test instance
            except Exception as e:
                print(f"    ❌ Failed to create target: {e}")
                continue
            
            for run_idx in tqdm(range(n_rerun), desc=f"{prob_name} runs"):
                print(f"\n    🔄 Run {run_idx + 1}/{n_rerun}")
                
                try:
                    print(f"        🧹 Clearing JAX cache...")
                    # Clear cache before each run
                    clear_jax_cache()
                    
                    print(f"        🎲 Splitting random keys...")
                    trainer_key, init_key, key = jax.random.split(key, 3)
                    print(f"        ✅ Keys created")
                    
                    print(f"        🎯 Creating target problem...")
                    ids = {}
                    target = problem()
                    print(f"        ✅ Target: {type(target).__name__}(dim={target.dim})")
                    
                    print(f"        📁 Creating output path...")
                    path = parent_path / f"{prob_name}"
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"        ✅ Path: {path}")

                    print(f"        🤖 Starting algorithm training...")
                    # Process algorithms sequentially to save memory
                    for algo_idx, algo in enumerate(algorithms_to_run):
                        print(f"            📋 Checking algorithm: {algo}")
                        if algo not in config:
                            print(f"            ⚠️  Skipping {algo} - not in config")
                            continue
                        
                        print(f"            🚀 Training {algo.upper()}...")
                        
                        try:
                            print(f"                🎲 Creating algorithm keys...")
                            m_key, key = jax.random.split(key, 2)
                            
                            print(f"                ⚙️  Creating parameters for {algo}...")
                            parameters = config_to_parameters(config, algo)
                            print(f"                ✅ Parameters created")
                            
                            print(f"                🔧 Creating step function and carry...")
                            step, carry = make_step_and_carry(init_key, parameters, target)
                            print(f"                ✅ Step function created")
                            
                            print(f"                🏃 Starting training chunks...")
                            # Train with periodic cache clearing
                            history_chunks = []
                            chunk_size = MEMORY_OPTIMIZED_CONFIG['clear_cache_freq']
                            
                            current_carry = carry
                            n_chunks = (n_updates + chunk_size - 1) // chunk_size
                            print(f"                    📊 Training in {n_chunks} chunks of {chunk_size}")
                            
                            for chunk_idx, chunk_start in enumerate(range(0, n_updates, chunk_size)):
                                chunk_end = min(chunk_start + chunk_size, n_updates)
                                chunk_updates = chunk_end - chunk_start
                                
                                print(f"                    🔄 Chunk {chunk_idx + 1}/{n_chunks}: {chunk_start}-{chunk_end} ({chunk_updates} updates)")
                                
                                chunk_key, trainer_key = jax.random.split(trainer_key, 2)
                                chunk_history, current_carry = trainer(
                                    chunk_key, current_carry, target, None, step,
                                    chunk_updates, metrics=None, use_jit=use_jit
                                )
                                history_chunks.append(chunk_history)
                                print(f"                    ✅ Chunk {chunk_idx + 1} completed")
                                
                                # Clear cache periodically
                                if chunk_end < n_updates:
                                    clear_jax_cache()
                            
                            print(f"                📈 Combining history...")
                            # Combine history chunks
                            history = defaultdict(list)
                            for chunk in history_chunks:
                                for k, v in chunk.items():
                                    history[k].extend(v)
                            
                            ids[algo] = current_carry.id
                            print(f"                ✅ {algo.upper()} training completed")
                            
                            # Store history
                            for k, v in history.items():
                                (path / f"{algo}").mkdir(exist_ok=True, parents=True)
                                histories[prob_name][algo][k].append(np.array(v))
                            
                            # Compute metrics with memory optimization
                            print(f"                📊 Computing metrics for {algo.upper()}...")
                            metrics = memory_safe_metrics_fn(m_key, target, ids[algo])
                            for met_key, met_value in metrics.items():
                                results[prob_name][algo][met_key].append(met_value)
                            
                            final_loss = history['loss'][-1] if 'loss' in history and history['loss'] else 0.0
                            print(f"                ✅ {algo.upper()} completed - Final loss: {final_loss:.4f}")
                            
                            # Clear algorithm-specific memory
                            del current_carry, history, step
                            clear_jax_cache()
                            
                        except Exception as e:
                            print(f"                ❌ Error training {algo.upper()}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                    # Create visualization (memory-safe)
                    try:
                        print(f"        🎨 Creating visualizations...")
                        visualize_key, key = jax.random.split(key, 2)
                        memory_safe_visualize(visualize_key, ids, target, path, prefix=f"run{run_idx}")
                        print(f"        ✅ Visualizations created")
                    except Exception as e:
                        print(f"        ⚠️  Visualization failed: {e}")
                    
                    # Clear run-specific memory
                    del ids, target
                    clear_jax_cache()
                    print(f"    ✅ Run {run_idx + 1} completed")
                    
                except Exception as e:
                    print(f"    ❌ Error in run {run_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate comparison results
    try:
        print("\n" + "="*80)
        print("MEMORY-OPTIMIZED T-PVI COMPARISON RESULTS")
        print("="*80)
        
        comparison_data = []
        for prob_name in problems_to_run.keys():
            if prob_name in results:
                row_data = {'Problem': prob_name.capitalize()}
                
                for algo in algorithms_to_run:
                    if algo in results[prob_name]:
                        power_runs = results[prob_name][algo]['power']
                        w_runs = results[prob_name][algo]['sliced_w']
                        
                        if len(power_runs) > 1:
                            power_mean = np.mean(power_runs)
                            power_std = np.std(power_runs)
                            w_mean = np.mean(w_runs)
                            w_std = np.std(w_runs)
                        else:
                            power_mean = power_runs[0] if power_runs else 0
                            power_std = 0
                            w_mean = w_runs[0] if w_runs else 0
                            w_std = 0
                        
                        row_data[f'{algo.upper()}_power'] = f"{power_mean:.3f}±{power_std:.3f}"
                        row_data[f'{algo.upper()}_wasserstein'] = f"{w_mean:.3f}±{w_std:.3f}"
                        
                        print(f"\n{prob_name.upper()} - {algo.upper()}:")
                        print(f"  Power (rejection rate): {power_mean:.3f} ± {power_std:.3f}")
                        print(f"  Wasserstein distance: {w_mean:.3f} ± {w_std:.3f}")
                
                comparison_data.append(row_data)
        
        # Save results
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        results = default_to_regular(results)
        histories = default_to_regular(histories)
        
        try:
            with open(parent_path / f'{name}_histories.pkl', 'wb') as f:
                pickle.dump(histories, f)
            with open(parent_path / f'{name}_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            # Save comparison table
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_csv(parent_path / f'{name}_comparison.csv', index=False)
                print(f"\n📊 Results saved to: {parent_path}")
                
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

        # Performance comparison
        if len(problems_to_run) > 0:
            print("\n" + "="*50)
            print("PERFORMANCE SUMMARY")
            print("="*50)
            
            for prob_name in problems_to_run.keys():
                if prob_name in results and 'pvi' in results[prob_name] and 'tpvi' in results[prob_name]:
                    pvi_power = np.mean(results[prob_name]['pvi']['power'])
                    tpvi_power = np.mean(results[prob_name]['tpvi']['power'])
                    pvi_w = np.mean(results[prob_name]['pvi']['sliced_w'])
                    tpvi_w = np.mean(results[prob_name]['tpvi']['sliced_w'])
                    
                    power_improvement = ((pvi_power - tpvi_power) / pvi_power) * 100 if pvi_power > 0 else 0
                    w_improvement = ((pvi_w - tpvi_w) / pvi_w) * 100 if pvi_w > 0 else 0
                    
                    print(f"\n{prob_name.upper()}:")
                    print(f"  T-PVI vs PVI improvement:")
                    print(f"    Power: {power_improvement:.1f}%")
                    print(f"    Wasserstein: {w_improvement:.1f}%")

        print(f"\n✅ Memory-optimized comparison completed!")
        print(f"💾 Results saved in: {parent_path}")

    except Exception as e:
        print(f"❌ Error during results generation: {e}")
        import traceback
        traceback.print_exc()


@app.command()
def test_memory():
    """Test script with minimal memory usage"""
    print("🧪 Testing memory-optimized T-PVI on Banana problem...")
    
    try:
        # Minimal test configuration
        key = jax.random.PRNGKey(42)
        target = Banana()
        
        # Very small test
        test_config = {
            'algorithm': 'tpvi',
            'model_parameters': ModelParameters(
                d_z=2, use_particles=True, n_particles=20, 
                kernel='norm_fixed_var_w_skip', n_hidden=64
            ),
            'theta_opt_parameters': ThetaOptParameters(lr=1e-4, optimizer='rmsprop'),
            'r_opt_parameters': ROptParameters(lr=1e-2, regularization=1e-8),
            'extra_alg_parameters': TPIDParameters(
                mc_n_samples=50, lambda_0=1e-2, lambda_min=1e-8
            )
        }
        
        parameters = Parameters(**test_config)
        step, carry = make_step_and_carry(key, parameters, target)
        
        # Short training run
        trainer_key, _ = jax.random.split(key, 2)
        history, final_carry = trainer(
            trainer_key, carry, target, None, step, 100, use_jit=True
        )
        
        print(f"✅ Test successful! Final loss: {history['loss'][-1]:.4f}")
        print("🎯 Ready to run full comparison with memory optimizations.")
        
        # Test T-PVI specific functionality
        if hasattr(final_carry, 'current_lambda_r'):
            print(f"🌡️  T-PVI annealing: λᵣ = {final_carry.current_lambda_r:.6f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Try reducing batch sizes or using CPU-only mode.")
        import traceback
        traceback.print_exc()


@app.command()
def debug():
    """Debug mode - check imports and basic functionality"""
    print("🔍 Debug Mode - Checking Dependencies")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("Testing imports...")
        import jax
        print(f"  ✅ JAX {jax.__version__}")
        
        import jax.numpy as np
        print("  ✅ JAX NumPy")
        
        from src.problems.toy import Banana
        print("  ✅ Toy problems")
        
        from src.base import TPIDParameters, Parameters, ModelParameters, ThetaOptParameters, ROptParameters
        print("  ✅ Base classes")
        
        from src.utils import make_step_and_carry, config_to_parameters
        print("  ✅ Utils")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        key = jax.random.PRNGKey(42)
        target = Banana()
        print(f"  ✅ Created Banana target (dim={target.dim})")
        
        # Test minimal T-PVI setup
        minimal_config = {
            'algorithm': 'tpvi',
            'model_parameters': ModelParameters(
                d_z=2, use_particles=True, n_particles=10, 
                kernel='norm_fixed_var_w_skip', n_hidden=32
            ),
            'theta_opt_parameters': ThetaOptParameters(lr=1e-4, optimizer='rmsprop'),
            'r_opt_parameters': ROptParameters(lr=1e-2, regularization=1e-8),
            'extra_alg_parameters': TPIDParameters(
                mc_n_samples=25, lambda_0=1e-2, lambda_min=1e-8
            )
        }
        
        parameters = Parameters(**minimal_config)
        step, carry = make_step_and_carry(key, parameters, target)
        print("  ✅ Created T-PVI step function and carry")
        
        # Test one training step
        trainer_key, _ = jax.random.split(key, 2)
        loss, new_carry = step(trainer_key, carry, target, None)
        print(f"  ✅ Single training step: loss = {loss:.4f}")
        
        if hasattr(new_carry, 'current_lambda_r'):
            print(f"  ✅ T-PVI annealing: λᵣ = {new_carry.current_lambda_r:.6f}")
        
        print("\n🎉 All checks passed! System is ready for T-PVI training.")
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\n💡 Suggestions:")
        print("   - Check that all dependencies are installed")
        print("   - Verify PYTHONPATH includes the project root")
        print("   - Try running: export PYTHONPATH=$PYTHONPATH:/content")


if __name__ == "__main__":
    app()