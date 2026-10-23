import os
import glob
import statistics
from collections import defaultdict

root_dir = '/Volumes/Files/Research/PhyNet-DTA/results'

def extract_fold_metrics(dir_path):
    metrics = defaultdict(list)
    folds = glob.glob(os.path.join(dir_path, 'test', 'fold_*', 'metrics.txt'))
    if not folds:
        return None
    for fold_file in folds:
        with open(fold_file, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    k, v = parts[0].strip(), float(parts[1].strip())
                    metrics[k].append(v)
    
    agg = {}
    for k, v in metrics.items():
        if len(v) > 1:
            agg[k] = f"{statistics.mean(v):.3f}±{statistics.stdev(v):.3f}"
        else:
            agg[k] = f"{statistics.mean(v):.3f}±0.000"
    return agg

def process_and_print():
    # Warm start
    print("=== Warm Start ===")
    davis_ws = extract_fold_metrics(os.path.join(root_dir, '5fold/results_5fold_davis/davis/setting_1'))
    print("Davis Setting 1:", davis_ws)
    kiba_ws = extract_fold_metrics(os.path.join(root_dir, '5fold/results_5fold_kiba/kiba/setting_1'))
    print("KIBA Setting 1:", kiba_ws)

    # Cold Start Davis
    print("\n=== Cold Start Davis ===")
    for s in ['setting_2', 'setting_3', 'setting_4']:
        print(s, extract_fold_metrics(os.path.join(root_dir, f'cold-start/results_cold_davis/davis/{s}')))

    # Cold Start KIBA
    print("\n=== Cold Start KIBA ===")
    for s in ['setting_2', 'setting_3', 'setting_4']:
        print(s, extract_fold_metrics(os.path.join(root_dir, f'cold-start/results_cold_kiba/kiba/{s}')))

    # Ablation Davis
    print("\n=== Ablation Davis ===")
    print("no_cross", extract_fold_metrics(os.path.join(root_dir, 'ab/davis/results_davis_no_cross/davis/setting_1')))
    print("no_phy", extract_fold_metrics(os.path.join(root_dir, 'ab/davis/results_davis_no_phy/davis/setting_1')))
    print("no_phy_cross", extract_fold_metrics(os.path.join(root_dir, 'ab/davis/results_davis_no_phy_cross/davis/setting_1')))

    # Ablation KIBA
    print("\n=== Ablation KIBA ===")
    print("no_cross", extract_fold_metrics(os.path.join(root_dir, 'ab/kiba/results_kiba_no_cross/kiba/setting_1')))
    print("no_phy", extract_fold_metrics(os.path.join(root_dir, 'ab/kiba/results_kiba_no_phy/kiba/setting_1')))
    print("no_phy_cross", extract_fold_metrics(os.path.join(root_dir, 'ab/kiba/results_kiba_no_phy_cross/kiba/setting_1')))

if __name__ == '__main__':
    process_and_print()
