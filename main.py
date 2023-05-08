import os
import argparse
import time
import numpy as np
import utils
from config import update_epochs_normalization
from algorithms.slad import SLAD


dataset_root = './'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--input_dir", type=str,
                    default='data',
                    help="the path of the data sets")
parser.add_argument("--output_dir", type=str,
                    default='&record/',
                    help="the output file path")
parser.add_argument("--dataset", type=str,
                    default='*thyroid,*arrhy*,*waveform',
                    help="FULL represents all the csv file in the folder, or a list of data set names split by comma")
parser.add_argument("--model", type=str, default='slad', help="",)
parser.add_argument("--normalization", type=str, default='min-max', help="",)
parser.add_argument('--contamination', type=float, default=-1)

parser.add_argument('--silent_header', action='store_true')
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')

# model parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--rep_dim', type=int, default=128)
parser.add_argument('--hidden_dims', type=str, default='100')
parser.add_argument('--d_size', type=int, default=10)
parser.add_argument('--n_ensemble', type=int, default=20)
parser.add_argument('--pool_size', type=int, default=50)
parser.add_argument('--mag_factor', type=int, default=200)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
data_lst = utils.get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)
print(os.path.join(dataset_root, args.input_dir))
print(data_lst)


model_class = SLAD
model_configs = {
    'epochs': args.epochs,
    'lr': args.lr,
    'distribution_size': args.d_size,
    'n_ensemble': args.n_ensemble,
    'hidden_dims': args.hidden_dims,
    'subspace_pool_size': args.pool_size,
    'n_unified_features': args.rep_dim,
    'magnify_factor': args.mag_factor
}
print('model configs:', model_configs)


cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
result_file = os.path.join(args.output_dir, f'{args.model}.{args.input_dir}.{args.flag}.csv')
raw_res_file = None


if not args.silent_header:
    f = open(result_file, 'a')
    print('\n---------------------------------------------------------', file=f)
    print(f'model: {args.model}, data dir: {args.input_dir}, '
          f'dataset: {args.dataset}, contamination: {args.contamination}, {args.runs}runs, ', file=f)
    print(f'{args.normalization}', file=f)
    for k in model_configs.keys():
        print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
    print(f'Note: {args.note}', file=f)
    print('---------------------------------------------------------', file=f)
    print('data, auc-roc, std, auc-pr, std, f1, std, time', file=f)
    f.close()


avg_auc_lst, avg_ap_lst, avg_f1_lst = [], [], []
for file in data_lst:
    dataset_name = os.path.splitext(os.path.split(file)[1])[0]

    print(f'\n-------------------------{dataset_name}-----------------------')

    # modify the normalization/epoch according to different datasets
    model_configs, normalization = update_epochs_normalization(args.model, dataset_name,
                                                               model_configs, args.normalization)

    print(f'normalization: {normalization}')
    x_train, y_train, x_test, y_test = utils.read_data(file=file,
                                                       normalization=normalization,
                                                       seed=42)
    if x_train is None:
        continue

    # # # Experiment: robustness w.r.t. different anomaly contamination rate
    if args.contamination != -1:
        x_train_, y_train_, x_test_, y_test_ = utils.adjust_contamination(
            x_train, y_train, x_test, y_test,
            contamination_r=args.contamination,
            swap_ratio=0.05,
            random_state=42
        )
    else:
        x_train_, y_train_, x_test_, y_test_ = x_train, y_train, x_test, y_test

    auc_lst, ap_lst, f1_lst = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    t1_lst, t2_lst = np.zeros(args.runs), np.zeros(args.runs)

    for i in range(args.runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        clf = model_class(**model_configs, random_state=42+i)
        clf.fit(x_train_)
        ckpt_time = time.time()
        scores = clf.decision_function(x_test_)
        done_time = time.time()

        auc, ap, f1 = utils.evaluate(y_test_, scores)
        auc_lst[i], ap_lst[i], f1_lst[i] = auc, ap, f1
        t1_lst[i] = ckpt_time - start_time
        t2_lst[i] = done_time - start_time

        print(f'{dataset_name}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {f1_lst[i]:.4f} '
              f'{t1_lst[i]:.1f}/{t2_lst[i]:.1f}, {model_configs}')

    avg_auc, avg_ap, avg_f1 = np.average(auc_lst), np.average(ap_lst), np.average(f1_lst)
    std_auc, std_ap, std_f1 = np.std(auc_lst), np.std(ap_lst), np.std(f1_lst)
    avg_time1 = np.average(t1_lst)
    avg_time2 = np.average(t2_lst)

    f = open(result_file, 'a')
    txt = f'{dataset_name}, ' \
          f'{avg_auc:.4f}, {std_auc:.4f}, ' \
          f'{avg_ap:.4f}, {std_ap:.4f}, ' \
          f'{avg_f1:.4f}, {std_f1:.4f}, ' \
          f'{avg_time1:.1f}/{avg_time2:.1f}, ' \
          f'norm, {normalization}, ' \
          f'{model_configs}'
    print(txt, file=f)
    print(txt)
    f.close()

    avg_auc_lst.append(avg_auc)
    avg_ap_lst.append(avg_ap)
    avg_f1_lst.append(avg_f1)


