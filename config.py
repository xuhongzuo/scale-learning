
epoch_table = {
    'slad': {
        '01-thyroid': 10, '02-arrhythmia': 10, '02-waveform': 10,
        '03-UNSW_NB15_traintest_DoS': 100,
        '05-bank': 100,
        '06-thrombin': 100,
        '27_PageBlocks': 100,
        'amazon': 100,
        'yelp': 100,
    },

}


def update_epochs_normalization(model, dataset_name, model_configs, normalization):
    # modify the normalization/epoch according to datasets

    if 'MVTec' in dataset_name:
        normalization = 'none'
        print(f'normalization update to: {normalization}')

    try:
        e = epoch_table[model][dataset_name]
        model_configs['epochs'] = e
        print(f'epochs update to: {e}')
    except KeyError:
        pass
    return model_configs, normalization

