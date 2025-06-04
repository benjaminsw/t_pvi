# create_visualizations.py
# Comprehensive visualization script for T-PVI vs PVI comparison

import matplotlib.pyplot as plt
import jax
from jax import vmap
import jax.numpy as np
from src.problems.toy import Banana, Multimodal, XShape
from src.utils import make_step_and_carry, config_to_parameters, parse_config
from src.trainers.trainer import trainer
from src.trainers.t_pvi import get_annealing_diagnostics
import numpy
from pathlib import Path
import pickle

def create_density_comparison():
    """Create side-by-side density plots comparing PVI vs T-PVI"""
    print("📊 Creating density comparison plots...")
    
    try:
        config = parse_config('scripts/sec_5.2/config/toy-tpvi-comparison.yaml')
    except FileNotFoundError:
        print("❌ Configuration file not found! Run setup cells first.")
        return
    
    key = jax.random.PRNGKey(42)
    problems = {
        'Banana': Banana(), 
        'Multimodal': Multimodal(), 
        'XShape': XShape()
    }
    algorithms = ['pvi', 'tpvi']
    colors = ['blue', 'red']
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(problems), 3, figsize=(18, 6*len(problems)))
    if len(problems) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (prob_name, target) in enumerate(problems.items()):
        print(f"  Processing {prob_name}...")
        
        # Create evaluation grid
        _max, _min = 4.5, -4.5
        x_lin = np.linspace(_min, _max, 200)
        XX, YY = np.meshgrid(x_lin, x_lin)
        XY = np.stack([XX, YY], axis=-1)
        
        # Compute true distribution
        log_p = lambda x: target.log_prob(x, None)
        log_true_ZZ = vmap(vmap(log_p))(XY)
        true_density = np.exp(log_true_ZZ)
        
        # Plot true distribution
        axes[i, 0].contour(XX, YY, true_density, levels=8, colors='black', linewidths=2)
        axes[i, 0].contourf(XX, YY, true_density, levels=20, alpha=0.6, cmap='Greys')
        axes[i, 0].set_title(f'{prob_name} - True Distribution', fontsize=14, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_aspect('equal')
        
        # Train and plot each algorithm
        for j, algo in enumerate(algorithms):
            trainer_key, init_key, key = jax.random.split(key, 3)
            
            # Quick training for visualization
            parameters = config_to_parameters(config, algo)
            step, carry = make_step_and_carry(init_key, parameters, target)
            
            # Reduced training for faster visualization
            n_updates = min(1500, config['experiment']['n_updates'])
            history, final_carry = trainer(
                trainer_key, carry, target, None, step, n_updates, use_jit=True)
            
            # Compute learned distribution
            model_log_p = lambda x: final_carry.id.log_prob(x, None)
            log_model_ZZ = vmap(vmap(model_log_p))(XY)
            model_density = np.exp(log_model_ZZ)
            
            # Plot learned distribution with true distribution overlay
            axes[i, j+1].contour(XX, YY, true_density, levels=8, 
                               colors='black', linewidths=1, alpha=0.5, linestyles='--')
            axes[i, j+1].contour(XX, YY, model_density, levels=8, 
                               colors=colors[j], linewidths=2)
            axes[i, j+1].contourf(XX, YY, model_density, levels=20, 
                                alpha=0.6, cmap='Blues' if j == 0 else 'Reds')
            
            # Add final loss to title
            final_loss = history['loss'][-1] if 'loss' in history else 0.0
            axes[i, j+1].set_title(f'{prob_name} - {algo.upper()} (Loss: {final_loss:.3f})', 
                                  fontsize=14, fontweight='bold')
            axes[i, j+1].set_xticks([])
            axes[i, j+1].set_yticks([])
            axes[i, j+1].set_aspect('equal')
            
            print(f"    {algo.upper()}: Final loss = {final_loss:.4f}")
    
    plt.tight_layout()
    plt.savefig('tpvi_vs_pvi_density_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("✅ Density comparison saved as 'tpvi_vs_pvi_density_comparison.png'")

def create_annealing_diagnostics():
    """Create comprehensive T-PVI annealing diagnostics"""
    print("🌡️  Creating T-PVI annealing diagnostics...")
    
    try:
        config = parse_config('scripts/sec_5.2/config/toy-tpvi-comparison.yaml')
    except FileNotFoundError:
        print("❌ Configuration file not found! Run setup cells first.")
        return
    
    key = jax.random.PRNGKey(42)
    
    # Use Banana problem for detailed diagnostics
    target = Banana()
    parameters = config_to_parameters(config, 'tpvi')
    
    trainer_key, init_key = jax.random.split(key, 2)
    step, carry = make_step_and_carry(init_key, parameters, target)
    
    # Store diagnostics during training
    diagnostics_history = []
    loss_history = []
    
    def diagnostic_step(key, carry, target, y):
        """Custom step function that collects diagnostics"""
        lval, new_carry = step(key, carry, target, y)
        
        if hasattr(new_carry, 'iteration'):  # T-PVI carry
            diag = get_annealing_diagnostics(new_carry)
            diagnostics_history.append(diag.copy())
        
        loss_history.append(float(lval))
        return lval, new_carry
    
    print("  Training T-PVI with diagnostic collection...")
    # Train with diagnostics (no JIT for diagnostic collection)
    n_updates = 3000
    pbar_key = trainer_key
    carry_current = carry
    
    from tqdm import tqdm
    for i in tqdm(range(n_updates), desc="Training with diagnostics"):
        step_key, pbar_key = jax.random.split(pbar_key, 2)
        loss, carry_current = diagnostic_step(step_key, carry_current, target, None)
    
    if not diagnostics_history:
        print("❌ No diagnostics collected. T-PVI might not be working correctly.")
        return
    
    print(f"  Collected {len(diagnostics_history)} diagnostic points")
    
    # Create comprehensive diagnostic plots
    fig = plt.figure(figsize=(16, 12))
    
    # Extract diagnostic data
    iterations = [d['iteration'] for d in diagnostics_history]
    lambda_rs = [d['current_lambda_r'] for d in diagnostics_history]
    entropies = [d['current_entropy'] for d in diagnostics_history]
    diversities = [d['current_diversity'] for d in diagnostics_history]
    annealing_stopped = [d['annealing_stopped'] for d in diagnostics_history]
    
    # 1. Temperature annealing schedule
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(iterations, lambda_rs, 'b-', linewidth=2, label='λᵣ(t)')
    plt.ylabel('Regularization Strength λᵣ', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.title('Temperature Annealing Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add annealing schedule info
    schedule = parameters.extra_alg_parameters.annealing_schedule
    lambda_0 = parameters.extra_alg_parameters.lambda_0
    plt.text(0.05, 0.95, f'Schedule: {schedule}\nλ₀: {lambda_0}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Particle entropy evolution
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(iterations, entropies, 'g-', linewidth=2, label='Entropy')
    plt.ylabel('Particle Entropy', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.title('Particle Entropy Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add entropy threshold line
    entropy_threshold = parameters.extra_alg_parameters.entropy_threshold
    plt.axhline(y=entropy_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'Threshold ({entropy_threshold})')
    plt.legend()
    
    # 3. Particle diversity evolution
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(iterations, diversities, 'r-', linewidth=2, label='Diversity')
    plt.ylabel('Particle Diversity', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.title('Particle Diversity Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add diversity threshold line
    diversity_threshold = parameters.extra_alg_parameters.diversity_threshold
    plt.axhline(y=diversity_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Threshold ({diversity_threshold})')
    plt.legend()
    
    # 4. Training loss evolution
    ax4 = plt.subplot(2, 3, 4)
    loss_iterations = np.arange(len(loss_history))
    plt.plot(loss_iterations, loss_history, 'purple', linewidth=2, label='ELBO Loss')
    plt.ylabel('ELBO Loss', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.title('Training Loss Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Annealing status
    ax5 = plt.subplot(2, 3, 5)
    stopped_iterations = [it for it, stopped in zip(iterations, annealing_stopped) if stopped]
    if stopped_iterations:
        stop_point = min(stopped_iterations)
        plt.axvline(x=stop_point, color='red', linewidth=3, alpha=0.8, 
                   label=f'Annealing stopped at {stop_point}')
        plt.text(stop_point + 100, 0.5, f'Stopped\nat {stop_point}', 
                rotation=90, verticalalignment='center')
    
    plt.plot(iterations, [1 if not stopped else 0 for stopped in annealing_stopped], 
             'o-', linewidth=2, markersize=3, label='Annealing active')
    plt.ylabel('Annealing Status', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.title('Annealing Control', fontsize=14, fontweight='bold')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Stopped', 'Active'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Multi-metric summary
    ax6 = plt.subplot(2, 3, 6)
    
    # Normalize metrics for comparison
    norm_lambda = np.array(lambda_rs) / max(lambda_rs)
    norm_entropy = np.array(entropies) / max(entropies) if max(entropies) > 0 else np.zeros_like(entropies)
    norm_diversity = np.array(diversities) / max(diversities) if max(diversities) > 0 else np.zeros_like(diversities)
    
    plt.plot(iterations, norm_lambda, 'b-', linewidth=2, alpha=0.8, label='λᵣ (norm)')
    plt.plot(iterations, norm_entropy, 'g-', linewidth=2, alpha=0.8, label='Entropy (norm)')
    plt.plot(iterations, norm_diversity, 'r-', linewidth=2, alpha=0.8, label='Diversity (norm)')
    
    plt.ylabel('Normalized Values', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.title('Multi-Metric Summary', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tpvi_annealing_diagnostics.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n📈 T-PVI Annealing Summary:")
    print(f"  Initial λᵣ: {lambda_rs[0]:.6f}")
    print(f"  Final λᵣ: {lambda_rs[-1]:.6f}")
    print(f"  λᵣ reduction: {(lambda_rs[0] - lambda_rs[-1]) / lambda_rs[0] * 100:.1f}%")
    print(f"  Final entropy: {entropies[-1]:.3f}")
    print(f"  Final diversity: {diversities[-1]:.3f}")
    print(f"  Annealing stopped: {'Yes' if annealing_stopped[-1] else 'No'}")
    if stopped_iterations:
        print(f"  Stopped at iteration: {min(stopped_iterations)}")
    
    print("✅ Annealing diagnostics saved as 'tpvi_annealing_diagnostics.png'")

def create_particle_evolution():
    """Visualize particle evolution during T-PVI training"""
    print("🔄 Creating particle evolution visualization...")
    
    try:
        config = parse_config('scripts/sec_5.2/config/toy-tpvi-comparison.yaml')
    except FileNotFoundError:
        print("❌ Configuration file not found! Run setup cells first.")
        return
    
    key = jax.random.PRNGKey(42)
    target = Banana()  # Use Banana for clear visualization
    parameters = config_to_parameters(config, 'tpvi')
    
    trainer_key, init_key = jax.random.split(key, 2)
    step, carry = make_step_and_carry(init_key, parameters, target)
    
    # Collect particle positions at different stages
    snapshots = []
    snapshot_iterations = [0, 200, 500, 1000, 2000]
    
    current_carry = carry
    current_key = trainer_key
    
    print("  Collecting particle snapshots during training...")
    for i in range(max(snapshot_iterations) + 1):
        if i in snapshot_iterations:
            snapshots.append({
                'iteration': i,
                'particles': current_carry.id.particles.copy(),
                'lambda_r': current_carry.current_lambda_r if hasattr(current_carry, 'current_lambda_r') else 0
            })
        
        if i < max(snapshot_iterations):
            step_key, current_key = jax.random.split(current_key, 2)
            _, current_carry = step(step_key, current_carry, target, None)
    
    # Create particle evolution plot
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4*len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]
    
    # Create background density plot
    x_lin = np.linspace(-6, 6, 100)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)
    log_p = lambda x: target.log_prob(x, None)
    log_true_ZZ = vmap(vmap(log_p))(XY)
    true_density = np.exp(log_true_ZZ)
    
    for i, (ax, snapshot) in enumerate(zip(axes, snapshots)):
        # Plot true density as background
        ax.contour(XX, YY, true_density, levels=8, colors='gray', alpha=0.5, linewidths=1)
        
        # Plot particles
        particles = snapshot['particles']
        ax.scatter(particles[:, 0], particles[:, 1], 
                  c='red', alpha=0.7, s=20, edgecolors='black', linewidths=0.5)
        
        # Styling
        iter_num = snapshot['iteration']
        lambda_r = snapshot.get('lambda_r', 0)
        ax.set_title(f'Iteration {iter_num}\nλᵣ = {lambda_r:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_ylabel('Particle Position (dim 2)', fontsize=10)
        ax.set_xlabel('Particle Position (dim 1)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('tpvi_particle_evolution.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("✅ Particle evolution saved as 'tpvi_particle_evolution.png'")

def create_performance_comparison():
    """Create performance comparison charts from saved results"""
    print("📊 Creating performance comparison from saved results...")
    
    # Try to load saved results
    results_path = Path("output/sec_5.2/tpvi_comparison_colab/detailed_results.pkl")
    
    if not results_path.exists():
        print("⚠️  No saved results found. Running quick comparison...")
        # Run a quick comparison to get some data
        from run_tpvi_comparison import run_comparison
        results, _ = run_comparison()
        if results is None:
            print("❌ Could not generate comparison data")
            return
    else:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    
    # Create comprehensive performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    problems = list(results.keys())
    algorithms = ['pvi', 'tpvi']
    metrics = ['power', 'sliced_w', 'loss', 'time']
    metric_names = ['Rejection Rate', 'Wasserstein Distance', 'Final Loss', 'Training Time (s)']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        x_pos = np.arange(len(problems))
        width = 0.35
        
        pvi_values = []
        tpvi_values = []
        pvi_stds = []
        tpvi_stds = []
        
        for problem in problems:
            if 'pvi' in results[problem] and metric in results[problem]['pvi']:
                pvi_vals = results[problem]['pvi'][metric]
                pvi_values.append(np.mean(pvi_vals))
                pvi_stds.append(np.std(pvi_vals))
            else:
                pvi_values.append(0)
                pvi_stds.append(0)
                
            if 'tpvi' in results[problem] and metric in results[problem]['tpvi']:
                tpvi_vals = results[problem]['tpvi'][metric]
                tpvi_values.append(np.mean(tpvi_vals))
                tpvi_stds.append(np.std(tpvi_vals))
            else:
                tpvi_values.append(0)
                tpvi_stds.append(0)
        
        # Create bars
        bars1 = ax.bar(x_pos - width/2, pvi_values, width, yerr=pvi_stds,
                      label='PVI', alpha=0.8, capsize=5, color='skyblue', edgecolor='blue')
        bars2 = ax.bar(x_pos + width/2, tpvi_values, width, yerr=tpvi_stds,
                      label='T-PVI', alpha=0.8, capsize=5, color='lightcoral', edgecolor='red')
        
        ax.set_xlabel('Problem', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.capitalize() for p in problems])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars, values, stds in [(bars1, pvi_values, pvi_stds), (bars2, tpvi_values, tpvi_stds)]:
            for bar, val, std in zip(bars, values, stds):
                if val > 0:  # Only label non-zero values
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('tpvi_performance_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance comparison saved as 'tpvi_performance_comparison.png'")

def main():
    """Main function to run all visualizations"""
    print("🎨 T-PVI Visualization Suite")
    print("=" * 50)
    
    # Create output directory
    Path("visualizations").mkdir(exist_ok=True)
    
    try:
        # 1. Density comparison
        create_density_comparison()
        print()
        
        # 2. Annealing diagnostics
        create_annealing_diagnostics()
        print()
        
        # 3. Particle evolution
        create_particle_evolution()
        print()
        
        # 4. Performance comparison
        create_performance_comparison()
        print()
        
        print("🎉 All visualizations completed successfully!")
        print("\nGenerated files:")
        print("  - tpvi_vs_pvi_density_comparison.png")
        print("  - tpvi_annealing_diagnostics.png") 
        print("  - tpvi_particle_evolution.png")
        print("  - tpvi_performance_comparison.png")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        print("Make sure you've run all setup cells and the comparison script first.")

if __name__ == "__main__":
    main()