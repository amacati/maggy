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

from __future__ import annotations

from types import SimpleNamespace, Type

from torch.nn import Module as TorchModule
from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel

from deepspeed.pipe import PipelineModule
from deepspeed.runtime.engine import DeepSpeedEngine
from fairscale.nn import FullyShardedDataParallel as FairscaleFullyShardedDataParallel


class MaggyDDPModuleWrapper(TorchDistributedDataParallel):

    __module = None  # Avoid overwriting torch module

    def __init__(self, *args, **kwargs):
        model = self.__module(*args, **kwargs).cuda()
        super().__init__(model)

    @classmethod
    def config(cls, module: Type[TorchModule]) -> Type[MaggyDDPModuleWrapper]:
        cls.__module = module
        return cls


class MaggyFairScaleModuleWrapper(FairscaleFullyShardedDataParallel):

    __module = None

    def __init__(self, *args, **kwargs):
        model = self.__module(*args, **kwargs).cuda()
        super().__init__(model)

    @classmethod
    def config(cls, module: Type[TorchModule]) -> Type[MaggyFairScaleModuleWrapper]:
        cls.__module = module
        return cls


class MaggyDeepSpeedModuleWrapper(DeepSpeedEngine):

    __module = None
    config_params = None

    def __init__(self, *args, **kwargs):
        model = self.__module(*args, **kwargs)  # No .cuda() necessary for DeepSpeed
        ds_args = SimpleNamespace(local_rank=0)
        super().__init__(
            ds_args,
            model,
            model_parameters=model.parameters(),
            config_params=self.config_params,
        )

    @classmethod
    def config(
        cls, module: Type[TorchModule], config_params: dict
    ) -> Type[MaggyDeepSpeedModuleWrapper]:
        assert (
            module == PipelineModule
        ), """Maggy currently doesn't support pipeline
             modules with DeepSpeed ZeRO."""
        cls.__module = module
        cls.config_params = config_params
        return cls
