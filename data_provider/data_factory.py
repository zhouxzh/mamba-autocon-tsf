from data_provider.data_loader import (
    Dataset_Array,
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
)
from torch.utils.data import DataLoader
from data_provider.samplers import ACFLagPairBatchSampler
from utils.acf_relation_contract import resolve_acf_relation_batching

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'electricity': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'national_illness': Dataset_Custom,
    # Optional dense numeric benchmarks.  These inputs have no CSV date
    # column, so Dataset_Array validates their declared npz/txt schema and
    # generates only the regular marks required by the embedding interface.
    'solar': Dataset_Array,
    'pems': Dataset_Array,
    'pems03': Dataset_Array,
    'pems04': Dataset_Array,
    'pems07': Dataset_Array,
    'pems08': Dataset_Array,
    'metr_la': Dataset_Array,
    'pems_bay': Dataset_Array,
}


def _record_acf_relation_batching(args, decision):
    """Attach the resolved sampler decision to the mutable run namespace."""

    args.mamba_acf_relation_batching_requested = decision['requested']
    args.mamba_acf_relation_batching_effective = decision['effective']
    args.mamba_acf_relation_batching_reason = decision['reason']
    args.mamba_acf_relation_selected_lag_support = bool(
        decision['selected_lag_support']
    )


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            if args.model == 'TimesNet':
                batch_size = 1  # bsz=1 for evaluation
            else:
                batch_size = args.batch_size
        freq = args.freq
    else:
        # Keep every validation window so short splits (e.g. ILI) still
        # produce a finite validation loss. Whether training drops a final
        # partial batch is an explicit experiment setting.
        shuffle_flag = flag == 'train'
        drop_last = flag == 'train' and bool(getattr(args, 'drop_last_train', False))
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
    data_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        train_ratio=args.train_ratio,
    )
    if Data is Dataset_Array:
        data_kwargs.update(
            data_format=getattr(args, 'data_format', 'auto'),
            data_key=getattr(args, 'data_key', 'data'),
            data_feature_index=getattr(args, 'data_feature_index', 0),
            target_index=getattr(args, 'target_index', -1),
            data_interval_minutes=getattr(
                args,
                'data_interval_minutes',
                None,
            ),
            data_start_timestamp=getattr(args, 'data_start_timestamp', '2000-01-01'),
        )
    data_set = Data(**data_kwargs)
    print(flag, len(data_set))
    relation_decision = resolve_acf_relation_batching(args)
    if flag == 'train' and bool(getattr(args, 'mamba_acf_relation_loss', False)):
        _record_acf_relation_batching(args, relation_decision)
    relation_pairing = flag == 'train' and relation_decision['use_lag_pairs']
    if relation_pairing:
        # ``train_peaks`` is resolved after fitting the train-only ACF prior
        # during experiment construction.  Train against those exact lags,
        # rather than the requested fallback list, so the sampler and loss
        # have one auditable relation definition.
        lags = getattr(
            args,
            'mamba_acf_relation_effective_lags',
            getattr(
                args,
                'mamba_acf_effective_lags',
                getattr(args, 'mamba_acf_lags', [24, 168, 336]),
            ),
        )
        if isinstance(lags, str):
            lags = [int(item) for item in lags.replace(',', ' ').split() if item]
        batch_sampler = ACFLagPairBatchSampler(
            data_set,
            batch_size=batch_size,
            lags=lags,
            drop_last=drop_last,
            seed=int(getattr(args, 'run_seed', getattr(args, 'seed', 0))),
            lag_balance=str(getattr(args, 'mamba_acf_relation_lag_balance', 'random')),
        )
        print('train ACF relation batching: lag_pairs={}, lag_balance={}'.format(
            list(batch_sampler.lags), batch_sampler.lag_balance
        ))
        return data_set, DataLoader(
            data_set,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
