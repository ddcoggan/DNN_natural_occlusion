# Created by David Coggan on 2025 01 06
import shutil
import os
import os.path as op
import pandas as pd
import numpy as np

from . import MODEL_BASE, BENCHMARK_DIR

def load_benchmark_scores(model_dir, benchmark, overwrite=False):

    results_dir = op.join(MODEL_BASE, model_dir, BENCHMARK_DIR)
    os.makedirs(results_dir, exist_ok=True)

    # manual fix for old results
    if op.isfile(f'{results_dir}/public_benchmarks/performance.csv'):
        results = pd.read_csv(
            f'{results_dir}/public_benchmarks/performance.csv')
        shutil.rmtree(f'{results_dir}/public_benchmarks')
        results.to_csv(f'{results_dir}/benchmark_scores.csv',
                       index=False)

    out_path = f'{results_dir}/benchmark_scores.csv'
    if not op.isfile(out_path):
        results = pd.DataFrame()
    else:
        results = pd.read_csv(out_path)
        if overwrite:
            results = results[results.benchmark != benchmark]

    return results, out_path

""""
# manual fixes for old / duplicated results
if 'accuracy' in results.columns:
    results['metric'] = 'accuracy'
    results = results.rename(columns={'accuracy': 'score'})
    results.to_csv(out_path, index=False)
for column in ['level_1', 'level_2', 'level_3']:
    if column not in results.columns:
        results[column] = None
if 'gla' in results.level_2.unique():
    level_2s = []
    for i, row in results.iterrows():
        if row.benchmark == 'ImageNet-C':
            level_2 = row.path.split('/')[2]
        else:
            level_2 = row.level_2
        level_2s.append(level_2)
    results.level_2 = level_2s
    results.to_csv(out_path, index=False)
if 'score.1' in results.columns:
    import numpy as np
    results.score = [results.score[i] if np.isfinite(results.score[
        i]) else results['score.1'][i] for i in range(len(
        results))]
    results.drop(columns='score.1', inplace=True)
    results.to_csv(out_path, index=False)
if 'val' in results.level_2.unique():
    results.level_1.replace('ILSVRC2012', 'val', inplace=True)
    import numpy as np
    results.level_2.replace('val', np.nan, inplace=True)
    results.to_csv(out_path, index=False)
if any(type(i) != float and 'tensor' in i for i in results.score):
    scores = []
    for i, row in results.iterrows():
        if 'tensor' in row.score:
            scores.append(float(row.score[7:-1]))
        else:
            scores.append(float(row.score))
    results.score = scores
    results.to_csv(out_path, index=False)
results.level_2 = [str(i)[:3] if '.' in str(i) else i for i in (
            results.level_2)]
results = results[results.level_1 != 'barVert12']

# remove duplicate values
results = results.drop_duplicates(
    subset=['benchmark', 'level_1', 'level_2', 'level_3', 'metric'], 
    keep='last')
results.to_csv(out_path, index=False)
"""