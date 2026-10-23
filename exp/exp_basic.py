import os
import torch
from models import (
    Autoformer,
    Transformer,
    TimesNet,
    TimeMixer,
    Nonstationary_Transformer,
    DLinear,
    NLinear,
    FEDformer,
    Informer,
    LightTS,
    ETSformer,
    Pyraformer,
    PatchTST,
    MICN,
    Crossformer,
    FiLM,
    AutoConNet,
    AutoConCI,
    iTransformer,
    GraphDLinear,
    STGCN,
    GraphWaveNet,
    DCRNN,
    AGCRN,
    RLinear,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'TimeMixer': TimeMixer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'GraphDLinear': GraphDLinear,
            'STGCN': STGCN,
            'GraphWaveNet': GraphWaveNet,
            'DCRNN': DCRNN,
            'AGCRN': AGCRN,
            'NLinear': NLinear,
            'RLinear': RLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'AutoConNet': AutoConNet,
            'AutoConCI': AutoConCI,
            'iTransformer': iTransformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    @staticmethod
    def trainable_parameters(model):
        """Return parameters enabled for optimization.

        Optional modules may remain in the state dict for checkpoint
        compatibility while being frozen when the active forward path does
        not consume them.  Optimizers should receive only this filtered list.
        """

        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    @staticmethod
    def optimizer_parameter_groups(model, weight_decay):
        """Build optimizer groups while protecting tiny optional adapters.

        The variable-token GDD adapter is a low-rank residual whose internal
        weights can be driven to zero before its scalar gate opens.  Keeping
        that adapter out of weight decay avoids turning the initialization
        safeguard into an additional source of collapse.  All other trainable
        parameters retain the caller's regularization.
        """

        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "variable_gdd_mixer." in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def count_parameters(self, model):
        return sum(parameter.numel() for parameter in self.trainable_parameters(model))
