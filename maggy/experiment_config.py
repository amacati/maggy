#
#   Copyright 2021 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import typing
from typing import Union

from maggy import Searchspace
from maggy.earlystop import AbstractEarlyStop
from maggy.optimizer import AbstractOptimizer
from maggy.ablation.ablationstudy import AblationStudy
from maggy.ablation.ablator import AbstractAblator

if typing.TYPE_CHECKING:
    import torch


class LagomConfig:
    def __init__(self, name: str, description: str, hb_interval: int):
        self.name = name
        self.description = description
        self.hb_interval = hb_interval


class OptimizationConfig(LagomConfig):
    def __init__(
        self,
        num_trials: int,
        optimizer: Union[str, AbstractOptimizer],
        searchspace: Searchspace,
        optimization_key: str = "metric",
        direction: str = "max",
        es_interval: int = 1,
        es_min: int = 10,
        es_policy: Union[str, AbstractEarlyStop] = "median",
        name: str = "HPOptimization",
        description: str = "",
        hb_interval: int = 1,
    ):
        super().__init__(name, description, hb_interval)
        if not num_trials > 0:
            raise ValueError("Number of trials should be greater than zero!")
        self.num_trials = num_trials
        self.optimizer = optimizer
        self.optimization_key = optimization_key
        self.searchspace = searchspace
        self.direction = direction
        self.es_policy = es_policy
        self.es_interval = es_interval
        self.es_min = es_min


class AblationConfig(LagomConfig):
    def __init__(
        self,
        ablation_study: AblationStudy,
        ablator: Union[str, AbstractAblator] = "loco",
        direction: str = "max",
        name: str = "ablationStudy",
        description: str = "",
        hb_interval: int = 1,
    ):
        super().__init__(name, description, hb_interval)
        self.ablator = ablator
        self.ablation_study = ablation_study
        self.direction = direction


class DistributedConfig(LagomConfig):

    BACKENDS = ["ddp", "deepspeed"]

    def __init__(
        self,
        module: torch.nn.Module,
        train_set: Union[str, torch.util.data.Dataset],
        test_set: Union[str, torch.util.data.Dataset],
        backend: str = "ddp",
        mixed_precision: bool = False,
        name: str = "torchDist",
        hb_interval: int = 1,
        description: str = "",
        hparams: dict = None,
        zero_lvl: int = 0,
        deepspeed_config: dict = None,
    ):
        super().__init__(name, description, hb_interval)
        self.module = module
        self.train_set = train_set
        self.test_set = test_set
        if backend not in self.BACKENDS:
            raise ValueError(
                """Backend {} not supported by Maggy.
                 Supported types are: {}""".format(
                    backend, self.BACKENDS
                )
            )
        self.backend = backend
        self.mixed_precision = mixed_precision
        self.hparams = hparams if hparams else {}
        self.zero_lvl = zero_lvl
        self.ds_config = deepspeed_config
