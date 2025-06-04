# scripts/sec_5.2/run_tpvi_comparison.py
from tqdm import tqdm
import typer
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


app = typer.Typer()

PROBLEMS = {
    'banana': Banana,
    'multimodal': Multimodal,
    'xshape': XShape,
}

ALGORITHMS = ['tpvi', 'pvi'] #, 'sm', 'svi', 'uvi']  # Added tpvi

def visualize(key, 
              ids,
              target,
              path,
              prefix=""):
    _max = 4.5
    _min = -4.5
    x_lin = np.linspace(_min, _max, 1000)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)
    log_p = lambda x : target.log_prob(x, None)
    log_true_ZZ = vmap(vmap(log_p))(XY)
    plt.clf()
    
    # Special handling for PVI and T-PVI comparison
    if 'pvi' in ids and 'tpvi' in ids:
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # True distribution
        c_true = axes[0].contour(
            XX, YY, np.exp(log_true_ZZ),
            levels=5, colors='black', linewidths=3)
        axes[0].set_title('True Distribution')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # PVI
        model_log_p = lambda x: ids['pvi'].log_prob(x, None)
        log_model_ZZ = vmap(vmap(model_log_p))(XY)
        axes[1].contour(XX, YY, np.exp(log_true_ZZ), levels=5, colors='black', linewidths=2, alpha=0.5)
        axes[1].contour(XX, YY, np.exp(log_model_ZZ), levels=5, colors='blue', linewidths=2)
        axes[1].set_title('PVI')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # T-PVI
        model_log_p = lambda x: ids['tpvi'].log_prob(x, None)
        log_model_ZZ = vmap(vmap(model_log_p))(XY)
        axes[2].contour(XX, YY, np.exp(log_true_ZZ), levels=5, colors='black', linewidths=2, alpha=0.5)
        axes[2].contour(XX, YY, np.exp(log_model_ZZ), levels=5, colors='red', linewidths=2)
        axes[2].set_title('T-PVI')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(path / f"{prefix}_pvi_tpvi_comparison.pdf")
        plt.close()

    # Original visualization for individual algorithms
    for alg, id in ids.items():
        plt.clf()
        c_true = plt.contour(
            XX, YY, np.exp(log_true_ZZ),
            levels=5, colors='black', linewidths=3, alpha=0.7)
        
        model_log_p = lambda x: id.log_prob(x, None)
        log_model_ZZ = vmap(vmap(model_log_p))(XY)
        c_model = plt.contour(
            XX, YY, np.exp(log_model_ZZ),
            levels=c_true._levels,
            colors='deepskyblue' if alg == 'pvi' else 'red' if alg == 'tpvi' else 'green',
            linewidths=2)
        
        plt.title(f'{alg.upper()} vs True Distribution')
        plt.xticks([])
        plt.yticks([])
        
        (path / alg).mkdir(exist_ok=True, parents=True)
        plt.savefig(path / alg / f"{prefix}_pdf.pdf")
        plt.close()


def test(key, x, y):
    output = mmdfuse(x, y, key)
    return output


def compute_power(
        key,
        target,
        id,
        n_samples=500,
        n_retries=100):
    avg_rej = 0
    for _ in range(n_retries):
        m_key, t_key, test_key, key = jax.random.split(key, 4)
        model_samples = id.sample(m_key, n_samples, None)
        target_samples = target.sample(t_key, n_samples, None)
        avg_rej = avg_rej + test(
            test_key, model_samples, target_samples,
        )
    return avg_rej / n_retries


def compute_w1(key,
               target,
               id,
               n_samples=10000,
               n_retries=1):
    distance = 0
    for _ in range(n_retries):
        m_key, t_key, key = jax.random.split(key, 3)
        model_samples = id.sample(m_key, n_samples, None)
        target_samples = target.sample(t_key, n_samples, None)
        distance = distance + sliced_wasserstein_distance(
            numpy.array(model_samples), numpy.array(target_samples),
            n_projections=100,
        )
    return distance / n_retries


def metrics_fn(key,
            target,
            id):
    power = compute_power(
        key, target, id, n_samples=1000, n_retries=100
    ) 
    sliced_w = compute_w1(key,
                          target,
                          id,
                          n_samples=10000,
                          n_retries=10)
    return {'power': power,
            'sliced_w': sliced_w}


@app.command()
def run(config_name: str = "toy-tpvi-comparison",
        seed: int = 2):
    config_path = Path(f"scripts/sec_5.2/config/{config_name}.yaml")
    assert config_path.exists()
    config = parse_config(config_path)

    n_rerun = config['experiment']['n_reruns']
    n_updates = config['experiment']['n_updates']
    name = config['experiment']['name']
    name = 'default' if len(name) == 0 else name
    compute_metrics = config['experiment']['compute_metrics']
    use_jit = config['experiment']['use_jit']

    parent_path = Path(f"output/sec_5.2/{name}")
    key = jax.random.PRNGKey(seed)
    histories = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    results = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    # Store results for Table 1 comparison
    table1_results = defaultdict(lambda: defaultdict(list))

    for prob_name, problem in PROBLEMS.items():
        print(f"\n=== Running experiments for {prob_name.upper()} ===")
        for i in tqdm(range(n_rerun), desc=f"{prob_name} runs"):
            trainer_key, init_key, key = jax.random.split(key, 3)
            ids = {}
            target = problem()
            path = parent_path / f"{prob_name}"
            path.mkdir(parents=True, exist_ok=True)

            for algo in ALGORITHMS:
                if algo not in config:
                    print(f"Skipping {algo} - not in config")
                    continue
                    
                m_key, key = jax.random.split(key, 2)
                parameters = config_to_parameters(config, algo)
                step, carry = make_step_and_carry(
                    init_key,
                    parameters,
                    target)
                
                metrics = compute_w1 if compute_metrics else None
                history, carry = trainer(
                    trainer_key,
                    carry,
                    target,
                    None,
                    step,
                    n_updates,
                    metrics=metrics,
                    use_jit=use_jit,
                )
                ids[algo] = carry.id
                
                # Store loss history
                for k, v in history.items():
                    plt.clf()
                    plt.plot(v, label=k)
                    (path / f"{algo}").mkdir(exist_ok=True, parents=True)
                    plt.savefig(path / f"{algo}" / f"iter{i}_{k}.pdf")
                    histories[prob_name][algo][k].append(np.stack(v, axis=0))
                
                # Compute metrics for Table 1
                metrics = metrics_fn(
                    m_key,
                    target,
                    ids[algo])
                for met_key, met_value in metrics.items():
                    results[prob_name][algo][met_key].append(met_value)
                    table1_results[prob_name][algo].append({
                        'run': i,
                        'power': metrics['power'],
                        'sliced_w': metrics['sliced_w']
                    })

            visualize_key, key = jax.random.split(key, 2)
            visualize(visualize_key,
                      ids,
                      target,
                      path,
                      prefix=f"iter{i}")

    # Generate Table 1 comparison
    print("\n" + "="*80)
    print("TABLE 1: COMPARISON RESULTS (Format: rejection_rate/wasserstein_distance)")
    print("="*80)
    
    table1_df_data = []
    for prob_name in PROBLEMS.keys():
        row_data = {'Problem': prob_name.capitalize()}
        for algo in ['pvi', 'tpvi', 'svi', 'uvi', 'sm']:
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
                
                # Format like the paper: power/wasserstein with std as subscript
                result_str = f"{power_mean:.2f}_{power_std:.2f}/{w_mean:.2f}_{w_std:.2f}"
                row_data[algo.upper()] = result_str
                
                # Highlight if better than baseline
                if algo == 'tpvi' and 'pvi' in results[prob_name]:
                    pvi_power = np.mean(results[prob_name]['pvi']['power'])
                    pvi_w = np.mean(results[prob_name]['pvi']['sliced_w'])
                    if power_mean < pvi_power or w_mean < pvi_w:
                        row_data[algo.upper()] = "**" + result_str + "**"
        
        table1_df_data.append(row_data)
    
    table1_df = pd.DataFrame(table1_df_data)
    print(table1_df.to_string(index=False))
    
    # Save detailed results
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
            
        # Save Table 1 results as CSV
        table1_df.to_csv(parent_path / f'{name}_table1_results.csv', index=False)
        
    except Exception as e:
        print(f"Failed to dump results: {e}")

    # Print summary as in original paper format
    print("\n" + "="*80)
    print("DETAILED RESULTS SUMMARY")
    print("="*80)
    
    for prob_name in PROBLEMS.keys():
        print(f"\n{prob_name.upper()}:")
        for algo in ['PVI', 'TPVI', 'SVI', 'UVI', 'SM']:
            algo_lower = algo.lower()
            if algo_lower in results[prob_name]:
                power_runs = results[prob_name][algo_lower]['power']
                w_runs = results[prob_name][algo_lower]['sliced_w']
                
                if len(power_runs) > 1:
                    power_mean = np.mean(power_runs)
                    power_std = np.std(power_runs)
                    w_mean = np.mean(w_runs)
                    w_std = np.std(w_runs)
                    
                    print(f"  {algo}: {power_mean:.3f}±{power_std:.3f} / {w_mean:.3f}±{w_std:.3f}")
                    
                    # Compare T-PVI vs PVI
                    if algo == 'TPVI' and 'pvi' in results[prob_name]:
                        pvi_power = np.mean(results[prob_name]['pvi']['power'])
                        pvi_w = np.mean(results[prob_name]['pvi']['sliced_w'])
                        power_improvement = ((pvi_power - power_mean) / pvi_power) * 100
                        w_improvement = ((pvi_w - w_mean) / pvi_w) * 100
                        print(f"    → Improvement over PVI: {power_improvement:.1f}% (power), {w_improvement:.1f}% (Wasserstein)")


@app.command()
def analyze_results(results_path: str):
    """Analyze saved results and generate comparison plots"""
    path = Path(results_path)
    
    with open(path / 'tpvi_comparison_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    problems = list(results.keys())
    metrics = ['power', 'sliced_w']
    
    for i, metric in enumerate(metrics):
        for j, problem in enumerate(problems):
            ax = axes[i, j]
            
            algorithms = ['pvi', 'tpvi', 'svi', 'uvi', 'sm']
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            means = []
            stds = []
            labels = []
            
            for algo, color in zip(algorithms, colors):
                if algo in results[problem]:
                    values = results[problem][algo][metric]
                    if len(values) > 1:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                    else:
                        mean_val = values[0] if values else 0
                        std_val = 0
                    
                    means.append(mean_val)
                    stds.append(std_val)
                    labels.append(algo.upper())
            
            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                         color=colors[:len(labels)], alpha=0.7)
            
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Rejection Rate' if metric == 'power' else 'Wasserstein Distance')
            ax.set_title(f'{problem.capitalize()} - {metric.replace("_", " ").title()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            
            # Add value labels on bars
            for bar, mean_val, std_val in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(path / 'tpvi_comparison_analysis.pdf')
    plt.close()
    
    print("Analysis complete. Check 'tpvi_comparison_analysis.pdf' for results.")


if __name__ == "__main__":
    app()