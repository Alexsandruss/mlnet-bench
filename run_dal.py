from os import system as cmd


def run_linear_regression(dataset):
    with open(f'data/{dataset}_train.csv') as training_file:
        features = len(training_file.read().split('\n')[0].split(',')) - 1
    cmd(f'sed -i -e "s/DATASET/{dataset}/" dal-benchs/lin_reg.cpp')
    cmd(f'sed -i -e "s/FEATURES/{features}/" dal-benchs/lin_reg.cpp')

    cmd('g++ dal-benchs/lin_reg.cpp -std=c++11 -I$CONDA_PREFIX/include -Idal-benchs/utils -L$CONDA_PREFIX/lib -ltbb -ltbbmalloc -lonedal_core -lonedal_thread')
    cmd('./a.out')

    cmd(f'sed -i -e "s/{dataset}/DATASET/" dal-benchs/lin_reg.cpp')
    cmd(f'sed -i -e "s/{features}/FEATURES/" dal-benchs/lin_reg.cpp')


if __name__ == '__main__':
    run_linear_regression('abalone')
    run_linear_regression('year_prediction_msd')
