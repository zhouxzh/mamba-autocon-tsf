from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import (
    StreamingMetricAccumulator,
    inverse_scaled_arrays,
    metric,
    per_channel_metrics,
    shape_metric,
    write_streaming_artifacts,
)
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from torch.onnx import OperatorExportTypes
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(f'model parameters:{self.count_parameters(model)}')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        weight_decay = getattr(self.args, 'weight_decay', None)
        if weight_decay is None:
            weight_decay = 0.0
        model_optim = optim.Adam(
            self.optimizer_parameter_groups(self.model, weight_decay),
            lr=self.args.learning_rate,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _read_acf_runtime_diagnostics(self):
        """Read opt-in Mamba ACF diagnostics without changing predictions."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        reader = getattr(model, 'acf_runtime_diagnostics', None)
        if not callable(reader):
            return {}
        values = {}
        for key, value in reader().items():
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values[str(key)] = number
        return values

    def vali(self, vali_data, vali_loader, criterion):
        # Aggregate squared error over all validation elements. Averaging batch
        # means would give a smaller final batch a different weight.
        total_squared_error = 0.0
        total_elements = 0
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                residual = outputs - batch_y
                total_squared_error += float(torch.sum(residual * residual).item())
                total_elements += int(residual.numel())
        total_loss = (total_squared_error / total_elements
                      if total_elements else float('inf'))
        self.model.train()
        return total_loss

    def train(self, setting):
        logs = dict()
        logs['train_loss'] = []
        logs['val_loss'] = []

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            logs['train_loss'].append(float(train_loss))
            logs['val_loss'].append(float(vali_loss))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        logs['best_epoch'] = int(np.argmin(logs['val_loss']) + 1) if logs['val_loss'] else None
        logs['epochs_ran'] = len(logs['val_loss'])
        json.dump(logs, open(os.path.join(path, 'loss_logs.json'), 'w'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        stream_metrics = bool(getattr(self.args, 'stream_metrics', False))
        origin_stride = getattr(self.args, 'origin_stride', None) or self.args.pred_len
        accumulator = (
            StreamingMetricAccumulator(
                test_data.scaler,
                int(self.args.c_out if self.args.features != 'MS' else 1),
                int(self.args.pred_len),
                origin_stride=int(origin_stride),
                sample_limit=int(getattr(self.args, 'stream_sample_windows', 4)),
            )
            if stream_metrics else None
        )
        preds = [] if not stream_metrics else None
        trues = [] if not stream_metrics else None
        collect_diagnostics = bool(
            stream_metrics and getattr(
                self.args, 'mamba_collect_eval_acf_diagnostics', False
            )
        )
        diagnostic_sums = {}
        diagnostic_counts = {}
        test_root = getattr(self.args, 'test_results_dir', './test_results/')
        folder_path = os.path.join(test_root, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if collect_diagnostics:
                    for key, value in self._read_acf_runtime_diagnostics().items():
                        diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + value
                        diagnostic_counts[key] = diagnostic_counts.get(key, 0) + 1

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                if stream_metrics:
                    accumulator.update(pred, true)
                else:
                    preds.append(pred)
                    trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

        if stream_metrics:
            result_root = getattr(self.args, 'results_dir', './results/')
            result_dir = os.path.join(result_root, setting)
            payload = write_streaming_artifacts(result_dir, accumulator)
            if diagnostic_sums:
                payload['acf_diagnostics'] = {
                    key: diagnostic_sums[key] / diagnostic_counts[key]
                    for key in sorted(diagnostic_sums)
                    if diagnostic_counts.get(key, 0)
                }
                # ``write_streaming_artifacts`` has already written the
                # canonical compact payload.  Rewrite it only when explicit
                # diagnostics were requested, keeping the default path
                # byte-for-byte unchanged and avoiding hidden GPU syncs.
                with open(
                    os.path.join(result_dir, 'stream_metrics.json'),
                    'w', encoding='utf-8'
                ) as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
                    handle.write('\n')
            scaled = payload['scaled']
            original = payload['original']
            mae, mse, rmse = scaled['mae'], scaled['mse'], scaled['rmse']
            mape, mspe = scaled['mape'], scaled['mspe']
            print('streamed test windows:', payload['window_count'])
            print(f'mse:{mse}, mae:{mae}, mape:{mape}, mspe:{mspe} '
                  'dilate:0.0000000, Shapedtw:0.0000000, Temporaldtw:0.0000000')
            result_log = getattr(self.args, 'result_log', 'result_long_term_forecast.txt')
            with open(result_log, 'a', encoding='utf-8') as handle:
                handle.write(setting + "  \n")
                handle.write('mse:{}, mae:{}'.format(mse, mae))
                handle.write('\n\n')
            return mse, mae, mape, mspe, 0.0, 0.0, 0.0

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        result_root = getattr(self.args, 'results_dir', './results/')
        folder_path = os.path.join(result_root, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # dilate_e, shape_e, temporal_e = shape_metric(preds, trues)  # These metrics take a long time to calculate.
        dilate_e, shape_e,temporal_e = 0.0,  0.0,  0.0
        print(f'mse:{mse}, mae:{mae}, mape:{mape}, mspe:{mspe} dilate:{dilate_e:.7f}, Shapedtw:{shape_e:.7f}, Temporaldtw:{temporal_e:.7f}')
        result_log = getattr(self.args, 'result_log', 'result_long_term_forecast.txt')
        f = open(result_log, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        original_preds, original_trues = inverse_scaled_arrays(preds, trues, test_data.scaler)
        original_metrics = metric(original_preds, original_trues)
        channel_metrics = np.column_stack((
            per_channel_metrics(preds, trues),
            per_channel_metrics(original_preds, original_trues),
        ))
        np.save(folder_path + 'original_metrics.npy', np.asarray(original_metrics))
        np.save(folder_path + 'channel_metrics.npy', channel_metrics)
        if self.args.save:
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        # Optional: export ONNX for visualization
        if getattr(self.args, 'export_onnx', False):
            try:
                export_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                export_model.eval()
                # get a sample batch
                sample = next(iter(test_loader))
                _, sx, sy, sxm, sym = sample
                # use a single sample to keep ONNX small
                sx = sx[:1].float().to(self.device)
                sy = sy[:1].float().to(self.device)
                sxm = sxm[:1].float().to(self.device)
                sym = sym[:1].float().to(self.device)

                # build decoder input same as during test
                dec_inp_sample = torch.zeros_like(sy[:, -self.args.pred_len:, :]).float()
                dec_inp_sample = torch.cat([sy[:, :self.args.label_len, :], dec_inp_sample], dim=1).float().to(self.device)

                onnx_dir = os.path.join(self.args.checkpoints, setting)
                os.makedirs(onnx_dir, exist_ok=True)
                onnx_path = os.path.join(onnx_dir, 'model.onnx')

                input_names = ["batch_x", "batch_x_mark", "dec_inp", "batch_y_mark"]
                output_names = ["output"]
                dynamic_axes = {
                    "batch_x": {0: "batch", 1: "seq_len"},
                    "batch_x_mark": {0: "batch", 1: "seq_len"},
                    "dec_inp": {0: "batch", 1: "total_len"},
                    "batch_y_mark": {0: "batch", 1: "total_len"},
                    "output": {0: "batch"}
                }

                torch.onnx.export(
                    export_model,
                    (sx, sxm, dec_inp_sample, sym),
                    onnx_path,
                    opset_version=17,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                )
                print(f"Exported ONNX model to: {onnx_path}")
            except Exception as e:
                print(f"Failed to export ONNX: {e}")

        return mse, mae, mape, mspe, dilate_e, shape_e, temporal_e
