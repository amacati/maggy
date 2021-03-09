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

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer


class MaggyZeroAdam(ZeroRedundancyOptimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    ):
        optim = torch.optim.Adam(params, lr, betas, eps, weight_decay, amsgrad)
        super().__init__(optim)


class MaggyZeroSGD(ZeroRedundancyOptimizer):
    def __init__(
        self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False
    ):
        optim = torch.optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov)
        super().__init__(optim)
