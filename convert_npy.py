import pandas as pd
import numpy as np
from multiprocessing import Pool


def get_data(dataset):
    global regression_datasets
    regression = dataset in regression_datasets
    data = {}
    # print(f'{dataset} dataset:')
    for el in ['x_train', 'x_test', 'y_train', 'y_test']:
        data[el] = np.load(f'../../automl-exp/sklbench_data/{dataset}_{el}.npy', allow_pickle=True)
        if 'y' in el:
            uniques = np.unique(data[el])
            # print(f'\t{el} unique values:', len(uniques))

    data['train'] = np.concatenate([data['x_train'], data['y_train'].reshape(-1, 1)], axis=1)
    data['test'] = np.concatenate([data['x_test'], data['y_test'].reshape(-1, 1)], axis=1)

    train_data = pd.DataFrame(data['train'], columns=[f'f{i}' for i in range(data['train'].shape[1] - 1)] + ['target'])
    test_data = pd.DataFrame(data['test'], columns=[f'f{i}' for i in range(data['test'].shape[1] - 1)] + ['target'])

    for col in list(train_data.columns):
        if col == 'target' and not regression:
            train_data[col] = train_data[col].astype(np.int8)
            test_data[col] = test_data[col].astype(np.int8)
        else:
            train_data[col] = train_data[col].astype(np.float32)
            test_data[col] = test_data[col].astype(np.float32)

    # print('\tTraining shape:', train_data.shape)
    # print('\tTesting shape:', test_data.shape)

    train_data.to_csv(f'{dataset}_train.csv', index=False)
    test_data.to_csv(f'{dataset}_test.csv', index=False)


datasets = [
    'abalone', 'letters', 'skin_segmentation', 'codrnanorm', 'ijcnn',
    'a9a', 'klaverjas', 'sensit', 'covertype', 'gisette', 'mnist',
    'higgs', 'airline-ohe', 'epsilon', 'year_prediction_msd'
]
regression_datasets = ['abalone', 'year_prediction_msd']

with Pool(len(datasets)) as pool:
    res = pool.map(get_data, datasets)
    for el in res:
        print(el)

# for dataset in datasets:
#     get_data(dataset, dataset in regression_datasets)
