import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import subprocess
import os
import io


def convert_to_csv(data, filename, label='target'):
    data = pd.DataFrame(data, columns=[f'f{i}' for i in range(data.shape[1] - 1)] + ['target'])
    data[label] = data[label].astype(int)
    data.to_csv(filename, index=False)


def read_output_from_command(command, env):
    res = subprocess.run(command.split(' '), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding='utf-8', env=env)
    return res.stdout[:-1], res.stderr[:-1]


random_state = 42
np.random.seed(42)

n_samples_range = [10000, 20000]
n_features_range = [8, 32]
n_trees_range = [50, 200]
n_leaves_range = [64, 256]

default_result = None
optimized_result = None
for n_samples in n_samples_range:
    for n_features in n_features_range:
        for n_trees in n_trees_range:
            for n_leaves in n_leaves_range:
                x, y = make_classification(n_samples=2 * n_samples, n_features=n_features, n_informative=n_features // 2, random_state=random_state)
                data = np.concatenate([x, y.reshape(-1, 1)], axis=1)
                train_data, test_data = train_test_split(data, test_size=0.5, random_state=random_state)

                train_filename, test_filename = "synth_data_bin_train.csv", "synth_data_bin_test.csv"
                convert_to_csv(train_data, train_filename)
                convert_to_csv(test_data, test_filename)

                env_copy = os.environ.copy()
                default_stdout, default_stderr = read_output_from_command(f'dotnet run {train_filename} {test_filename} binary {n_trees} {n_leaves}', env_copy)

                print(f'n_samples={n_samples}, n_features={n_features}, n_trees={n_trees}, n_leaves={n_leaves}')
                print(default_stdout)
                print(default_stderr)
                default_res = io.StringIO(default_stdout + '\n')
                default_res = pd.read_csv(default_res) # .rename(columns={'all workflow time[ms]', 'training accuracy', 'testing accuracy', 'training F1 score', 'testing F1 score'})
                default_res['n samples'] = np.array([n_samples, ])
                default_res['n features'] = np.array([n_features, ])
                default_res['n trees'] = np.array([n_trees, ])
                default_res['n leaves'] = np.array([n_leaves, ])
                if default_result is None:
                    default_result = default_res.copy()
                else:
                    default_result = pd.concat([default_result, default_res], axis=0)

                env_copy['MLNET_BACKEND'] = 'ONEDAL'
                optimized_stdout, optimized_stderr = read_output_from_command(f'dotnet run {train_filename} {test_filename} binary {n_trees} {n_leaves}', env_copy)

                print(optimized_stdout)
                print(optimized_stderr)
                optimized_res = io.StringIO(optimized_stdout + '\n')
                optimized_res = pd.read_csv(optimized_res)
                optimized_res['n samples'] = np.array([n_samples, ])
                optimized_res['n features'] = np.array([n_features, ])
                optimized_res['n trees'] = np.array([n_trees, ])
                optimized_res['n leaves'] = np.array([n_leaves, ])
                if optimized_result is None:
                    optimized_result = optimized_res.copy()
                else:
                    optimized_result = pd.concat([optimized_result, optimized_res], axis=0)

default_result = default_result.rename(columns={el: f'ML.NET {el}' for el in ['all workflow time[ms]', 'training accuracy', 'testing accuracy', 'training F1 score', 'testing F1 score']})
optimized_result = optimized_result.rename(columns={el: f'oneDAL {el}' for el in ['all workflow time[ms]', 'training accuracy', 'testing accuracy', 'training F1 score', 'testing F1 score']})

default_result.to_csv('default_result.csv', index=False)
optimized_result.to_csv('optimized_result.csv', index=False)
