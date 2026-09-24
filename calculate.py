import numpy as np
import pandas as pd
from datetime import datetime
import argparse


start_time = datetime.now()
parser = argparse.ArgumentParser()

parser.add_argument('-m','--method', default=f'CutAddPaste', type=str,
                    help='Method name')
parser.add_argument('-ed','--experiment_description', default=f'Exp-{start_time.replace(minute=0, second=0, microsecond=0)}', type=str,
                    help='Experiment Description')
parser.add_argument('-d','--selected_dataset', default='WADI', type=str,
                    help='Dataset of choice: IOpsCompetition, UCR, NAB, SWaT, WADI')
parser.add_argument('-p','--path', default='./results', type=str,
                    help='path to calculate')
args = parser.parse_args()

result_path = f'{args.path}{args.method}_{args.selected_dataset}_summary_{args.experiment_description}'

df = pd.read_csv(f'{result_path}.csv')

# df.at[0,'Overall'] = df.loc[0, df.columns[1]]
rows_to_calculate = df.iloc[4:27].iloc[:,1:]
rows_to_calculate = rows_to_calculate.apply(lambda x: x.str.split(' '))
# print(rows_to_calculate)

rows_to_calculate = [[float(value[0]) for value in rows_to_calculate[sublist]] for sublist in rows_to_calculate]
# print(rows_to_calculate)

mean_values = np.mean(rows_to_calculate,axis=0)
std_values = np.std(rows_to_calculate,axis=0)

df.loc[4:27, 'mean'] = mean_values # mean列填充为各列均值的平均
df.loc[4:27, 'std'] = std_values # std列填充为各列方差的平均

df.to_csv(f'{result_path}-calculated.csv', index=False)