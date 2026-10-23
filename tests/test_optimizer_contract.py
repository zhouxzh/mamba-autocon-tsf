"""Optimizer parameter-group contracts for frozen optional model branches."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from exp.exp_basic import Exp_Basic
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_term_forecasting_with_AutoCon import (
    Exp_Long_Term_Forecast_with_AutoCon,
)


class OptimizerParameterContractTests(unittest.TestCase):
    def _model(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        for parameter in model[1].parameters():
            parameter.requires_grad_(False)
        return model

    def test_helper_filters_frozen_parameters(self):
        model = self._model()
        trainable = Exp_Basic.trainable_parameters(model)

        self.assertEqual(
            {id(parameter) for parameter in trainable},
            {id(parameter) for parameter in model[0].parameters()},
        )

    def test_plain_optimizer_excludes_frozen_parameters(self):
        model = self._model()
        experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        experiment.model = model
        experiment.args = SimpleNamespace(learning_rate=1e-3, weight_decay=None)

        optimizer = experiment._select_optimizer()
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in model[0].parameters()})

    def test_autocon_optimizer_excludes_frozen_parameters(self):
        model = self._model()
        experiment = Exp_Long_Term_Forecast_with_AutoCon.__new__(
            Exp_Long_Term_Forecast_with_AutoCon
        )
        experiment.model = model
        experiment.args = SimpleNamespace(learning_rate=1e-3, weight_decay=None)

        optimizer = experiment._select_optimizer()
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in model[0].parameters()})

    def test_gdd_adapter_uses_no_weight_decay_group(self):
        model = nn.Module()
        model.backbone = nn.Linear(3, 3)
        model.variable_gdd_mixer = nn.Linear(3, 3)
        experiment = Exp_Long_Term_Forecast_with_AutoCon.__new__(
            Exp_Long_Term_Forecast_with_AutoCon
        )
        experiment.model = model
        experiment.args = SimpleNamespace(learning_rate=1e-3, weight_decay=1e-5)

        optimizer = experiment._select_optimizer()
        by_parameter = {
            id(parameter): group["weight_decay"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(by_parameter[id(model.backbone.weight)], 1e-5)
        self.assertEqual(by_parameter[id(model.variable_gdd_mixer.weight)], 0.0)

    def test_autocon_wrapper_requests_repr_only_while_training(self):
        class _RepresentationModel(nn.Module):
            supports_return_repr_control = True

            def __init__(self):
                super().__init__()
                self.AutoCon = True
                self.calls = []

            def forward(self, *inputs, return_repr=None):
                self.calls.append(return_repr)
                prediction = torch.zeros(1, 2, 1)
                representation = torch.ones(1, 3, 4) if return_repr else None
                return prediction, representation

        model = _RepresentationModel()
        experiment = Exp_Long_Term_Forecast_with_AutoCon.__new__(
            Exp_Long_Term_Forecast_with_AutoCon
        )
        experiment.model = model.eval()
        inputs = (
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 1),
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
        )

        _, representation = experiment._forecast_output(*inputs)
        self.assertIsNone(representation)
        self.assertEqual(model.calls[-1], False)

        model.train()
        _, representation = experiment._forecast_output(*inputs)
        self.assertIsNotNone(representation)
        self.assertEqual(model.calls[-1], True)


if __name__ == "__main__":
    unittest.main()
