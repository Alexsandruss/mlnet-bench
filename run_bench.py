import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import subprocess
import sys
import os
import io


RANDOM_STATE = 42
DEFAULT_DATA_FOLDER = 'data'
np.random.seed(RANDOM_STATE)


def read_output_from_command(command, env):
    res = subprocess.run(command.split(' '), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding='utf-8', env=env)
    return res.stdout[:-1], res.stderr[:-1]


def convert_to_csv(data, filename, label='target'):
    data = pd.DataFrame(data, columns=[f'f{i}' for i in range(data.shape[1] - 1)] + ['target'])
    data[label] = data[label].astype(int)
    data.to_csv(filename, index=False)


def generate_synthetic_data(n_training_samples, n_testing_samples, n_features, n_classes=0, data_folder=DEFAULT_DATA_FOLDER):
    n_samples = n_training_samples + n_testing_samples
    data_params = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_features // 2,
        'random_state': RANDOM_STATE
    }
    if n_classes == 0:
        x, y = make_regression(**data_params, noise=1.0, bias=1.0)
    else:
        x, y = make_classification(**data_params, n_classes=n_classes)

    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    train_data, test_data = train_test_split(data, test_size=n_testing_samples / n_samples, random_state=RANDOM_STATE)
    train_filename, test_filename = (
        f"{data_folder}/synth_data_train_{n_training_samples}_of_{n_samples}_{n_features}_{n_classes if n_classes else 'reg'}.csv",
        f"{data_folder}/synth_data_test_{n_testing_samples}_of_{n_samples}_{n_features}_{n_classes if n_classes else 'reg'}.csv")
    convert_to_csv(train_data, train_filename)
    convert_to_csv(test_data, test_filename)

    return train_filename, test_filename


def read_dotnet_output(output):
    return pd.read_csv(io.StringIO(output))


def run_bench(env, training_filename, testing_filename, label_name, task, algorithm, *args):
    command = f'dotnet run {training_filename} {testing_filename} {label_name} {task} {algorithm}'
    for arg in args:
        command += f' {arg}'
    print(command)
    return read_output_from_command(command, env)


def run_case(training_filename, testing_filename, label_name, task, algorithm, *args, n_runs=5):
    higher_is_better_metrics = ['accuracy', 'F1 score', 'R2 score']
    if task == 'binary':
        quality_metrics = ['accuracy', 'F1 score']
    elif task == 'regression':
        quality_metrics = ['RMSE', 'R2 score']
    else:
        raise ValueError(f'unknown "{task}" task')
    quality_metrics = [f'training {m}' for m in quality_metrics] + [f'testing {m}' for m in quality_metrics]
    metrics = ['workload time[ms]'] + quality_metrics

    # default ML.NET run
    env_copy = os.environ.copy()
    default_result = None
    for i in range(n_runs):
        stdout, stderr = run_bench(env_copy, training_filename, testing_filename, label_name, task, algorithm, *args)
        print('DEFAULT STDOUT:', stdout, 'DEFAULT STDERR:', stderr, sep='\n')
        new_result = read_dotnet_output(stdout + '\n')
        if default_result is None:
            default_result = new_result
        else:
            default_result = pd.concat([default_result, new_result], axis=0)
    groupby_columns = list(set(default_result.columns) - set(metrics))
    default_result = default_result.groupby(groupby_columns)[metrics].mean().reset_index()

    # oneDAL-accelerated run
    env_copy['MLNET_BACKEND'] = 'ONEDAL'
    optim_result = None
    for i in range(n_runs):
        stdout, stderr = run_bench(env_copy, training_filename, testing_filename, label_name, task, algorithm, *args)
        print('OPTIMIZED STDOUT:', stdout, 'OPTIMIZED STDERR:', stderr, sep='\n')
        new_result = read_dotnet_output(stdout + '\n')
        if optim_result is None:
            optim_result = new_result
        else:
            optim_result = pd.concat([optim_result, new_result], axis=0)
    optim_result = optim_result.groupby(groupby_columns)[metrics].mean().reset_index()

    # comparison of results
    metrics_map_template = {el: '{} ' + f'{el}' for el in metrics}
    default_result = default_result.rename(columns={k: v.format('ML.NET') for k, v in metrics_map_template.items()})
    optim_result = optim_result.rename(columns={k: v.format('oneDAL') for k, v in metrics_map_template.items()})

    result = default_result.merge(optim_result)
    for metric in metrics:
        if metric.split(' ', 1)[-1] in higher_is_better_metrics:
            result[f'oneDAL/ML.NET {metric}'] = \
                result[f'oneDAL {metric}'] / result[f'ML.NET {metric}']
        else:
            result[f'ML.NET/oneDAL {metric}'.replace('[ms]', '')] = \
                result[f'ML.NET {metric}'] / result[f'oneDAL {metric}']
    return result


if __name__ == '__main__':
    # n_samples_range = [20000, 50000]
    # n_features_range = [16, 128]
    # n_trees_range = [100, 500]
    # n_leaves_range = [64, 256]

    # result = None
    # for n_samples in n_samples_range:
    #     for n_features in n_features_range:
    #         training_filename, testing_filename = generate_synthetic_data(n_samples, n_samples, n_features, n_classes=2)
    #         label_name = 'target'
    #         for n_trees in n_trees_range:
    #             for n_leaves in n_leaves_range:
    #                 new_result = run_case(training_filename, testing_filename, label_name, 'binary', 'RF', n_trees, n_leaves)
    #                 for k, v in {'n_leaves': n_leaves, 'n_trees': n_trees, 'n_features': n_features, 'n_samples': n_samples}.items():
    #                     new_result.insert(1, k, [v])
    #                 if result is None:
    #                     result = new_result
    #                 else:
    #                     result = pd.concat([result, new_result], axis=0)

    # result.to_csv('rfc_result.csv', index=False)
    # result.to_csv(sys.stdout, index=False)

    n_samples_range = [20000, 50000, 100000, 200000, 500000]
    n_features_range = [8, 16, 32, 64, 128, 256]
    n_iterations = 1000

    result = None
    for n_samples in n_samples_range:
        for n_features in n_features_range:
            training_filename, testing_filename = generate_synthetic_data(n_samples, n_samples, n_features, n_classes=2)
            label_name = 'target'
            new_result = run_case(training_filename, testing_filename, label_name, 'binary', 'LR', n_iterations)
            for k, v in {'n_features': n_features, 'n_samples': n_samples}.items():
                new_result.insert(1, k, [v])
            if result is None:
                result = new_result
            else:
                result = pd.concat([result, new_result], axis=0)
            os.system(f'rm {training_filename} {testing_filename}')

    result.to_csv('logreg_result.csv', index=False)
    result.to_csv(sys.stdout, index=False)

    result = None
    for n_samples in n_samples_range:
        for n_features in n_features_range:
            training_filename, testing_filename = generate_synthetic_data(n_samples, n_samples, n_features, n_classes=0)
            label_name = 'target'
            new_result = run_case(training_filename, testing_filename, label_name, 'regression', 'OLS')
            for k, v in {'n_features': n_features, 'n_samples': n_samples}.items():
                new_result.insert(1, k, [v])
            if result is None:
                result = new_result
            else:
                result = pd.concat([result, new_result], axis=0)
            os.system(f'rm {training_filename} {testing_filename}')

    result.to_csv('olsreg_result.csv', index=False)
    result.to_csv(sys.stdout, index=False)
