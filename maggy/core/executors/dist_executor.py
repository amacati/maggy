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

import builtins as __builtin__
import inspect
import traceback
import os
import datetime
import random
import socket
from typing import Callable, Union, Any, Tuple, Type

import torch
import torch.distributed as dist

import numpy as np
import deepspeed

from maggy import util
from maggy.experiment_config import DistributedConfig
from maggy.core.rpc import Client
from maggy.core.reporter import Reporter
from maggy.core.patching import (
    MaggyDataLoader,
    MaggyZeroAdam,
    MaggyZeroSGD,
    MaggyDDPModuleWrapper,
    MaggyFairScaleModuleWrapper,
    MaggyDeepSpeedModuleWrapper,
)
from maggy.core.environment.singleton import EnvSing


def dist_executor_fn(
    train_fn: Callable,
    config: DistributedConfig,
    app_id: int,
    run_id: int,
    server_addr: str,
    hb_interval: int,
    secret: str,
    log_dir: str,
) -> Callable:
    """Wraps the user supplied training function in order to be passed to the Spark Executors.

    :param train_fn: Original training function.
    :param config: Experiment config.
    :param app_id: Maggy application ID.
    :param run_id: Maggy run ID.
    :param server_addr: IP of the Maggy worker registration RPC server.
    :param hb_interval: Worker heartbeat interval.
    :param secret: Secret string to authenticate messages.
    :param log_dir: Location of the logger file directory on the file system.

    :returns: Patched function to execute on the Spark executors.
    """

    def wrapper_function(_: Any) -> None:
        """Patched function from dist_executor_fn factory.

        :param _: Necessary catch for the iterator given by Spark to the
        function upon foreach calls. Can safely be disregarded.
        """
        EnvSing.get_instance().set_ml_id(app_id, run_id)
        partition_id, _ = util.get_partition_attempt_id()
        client = Client(server_addr, partition_id, 0, hb_interval, secret)
        log_file = log_dir + "/executor_" + str(partition_id) + ".log"

        reporter = Reporter(log_file, partition_id, 0, __builtin__.print)
        builtin_print = __builtin__.print

        def maggy_print(*args, **kwargs):
            builtin_print(*args, **kwargs)
            reporter.log(" ".join(str(x) for x in args), True)

        __builtin__.print = maggy_print

        try:
            _register_with_servers(client, reporter, partition_id)
            tb_logdir, trial_log_file = _setup_logging(reporter, log_dir)
            reporter.log("Awaiting worker reservations.", True)
            client.await_reservations()
            reporter.log("Reservations complete, configuring PyTorch.", True)
            master_config = client.get_torch_config()
            if not master_config:
                reporter.log(
                    "PyTorch registration failed, exiting from all tasks.", True
                )
                return
            addr, port = master_config["host_port"].split(":")
            torch_config = {
                "MASTER_ADDR": addr,
                "MASTER_PORT": port,
                "WORLD_SIZE": str(master_config["num_executors"]),
                "RANK": str(partition_id),
                "LOCAL_RANK": str(0),  # DeepSpeed requires local rank.
                "NCCL_BLOCKING_WAIT": "1",
            }
            reporter.log(f"Torch config is {torch_config}")

            _setup_torch_env(torch_config)
            _init_cluster(timeout=60, random_seed=0)
            module = _wrap_module(config)
            _monkey_patch_pytorch(config.zero_lvl)

            reporter.log("Starting distributed training.", True)
            sig = inspect.signature(train_fn)
            if sig.parameters.get("reporter", None):
                retval = train_fn(
                    module=module,
                    hparams=config.hparams,
                    train_set=config.train_set,
                    test_set=config.test_set,
                    reporter=reporter,
                )
            else:
                retval = train_fn(
                    module=module,
                    hparams=config.hparams,
                    train_set=config.train_set,
                    test_set=config.test_set,
                )

            retval = util.handle_return_val(retval, tb_logdir, "Metric", trial_log_file)

            reporter.log("Finished distributed training.", True)
            reporter.log("Final metric: {}".format(retval), True)
            client.finalize_metric(retval, reporter)
        except:  # noqa: E722
            reporter.log(traceback.format_exc(), False)
            raise
        finally:
            reporter.close_logger()
            client.stop()
            client.close()

    return wrapper_function


def _register_with_servers(
    client: Client, reporter: Reporter, partition_id: int
) -> None:
    """Registers own address with server and starts heartbeat protocol.

    :param client: Client for communication with the server.
    :param reporter: Reporter responsible for heartbeat.
    :param partition_id: Executors partition ID from Sparks RDD.
    """
    client_addr = client.client_addr
    port = _get_open_port()
    host_port = client_addr[0] + ":" + str(port)
    exec_spec = {
        "partition_id": partition_id,
        "task_attempt": 0,
        "host_port": host_port,
        "trial_id": None,
    }
    client.register(exec_spec)
    client.start_heartbeat(reporter)


def _get_open_port() -> str:
    """Lets the OS choose a free socket and attempts to bind it.

    :returns: The port name.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to 0 lets OS choose a free socket.
    port = sock.getsockname()[1]
    sock.close()
    return port


def _setup_logging(reporter: Reporter, log_dir: str) -> Tuple[str, str]:
    """Sets up logging directories and files.

    :param reporter: Reporter responsible for logging.
    :param log_dir: Log directory path on the file system.

    :returns: Tuple containing the path of the tensorboard directory
        and the trial log file.
    """
    reporter.set_trial_id(0)
    tb_logdir = log_dir + "/" + "training_logs_" + str(reporter.partition_id)
    trial_log_file = tb_logdir + "/output.log"
    reporter.set_trial_id(0)
    # If trial is repeated, delete trial directory, except log file
    if EnvSing.get_instance().exists(tb_logdir):
        util.clean_dir(tb_logdir, [trial_log_file])
    else:
        EnvSing.get_instance().mkdir(tb_logdir)
    reporter.init_logger(trial_log_file)
    return tb_logdir, trial_log_file


def _setup_torch_env(torch_config: dict) -> None:
    """Registers the Torch config as environment variables on the worker.

    :param torch_config: Dictionary containing the values of the variables.
    """
    for env_variable in torch_config.keys():
        os.environ[env_variable] = str(torch_config[env_variable])


def _init_cluster(
    timeout: int = 60, random_seed: int = 0, backend: str = "ddp"
) -> None:
    """Checks if config is set, initializes the Torch distributed cluster and sets random seeds.

    :param timeout: Time until initialization times out (default: ``60``).
    :param random_seed: Random seed for Torch, numpy, random (default: ``0``).
    :param backend: The backend that torch uses for distributed training. Either "ddp"
        or "deepspeed" (default: ``ddp``).

    :raises KeyError: Checks on environment variables failed.
    :raises RuntimeError: Checks on PyTorch distributed backend failed.
    """
    for env_variable in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "NCCL_BLOCKING_WAIT",
    ]:
        if env_variable not in os.environ:
            raise KeyError(f"Environment variable {env_variable} not registered!")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch distributed needs a GPU cluster.")
    if not dist.is_available():
        raise RuntimeError("Torch distributed backend not accessible.")
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL link not available on worker.")
    if backend == "deepspeed":
        deepspeed.init_process_group(timeout=datetime.timedelta(seconds=timeout))
    else:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout)
        )
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def _wrap_module(
    config: DistributedConfig,
) -> Union[
    Type[MaggyDDPModuleWrapper],
    Type[MaggyFairScaleModuleWrapper],
    Type[MaggyDeepSpeedModuleWrapper],
]:
    """Wraps the module according to `backend`.

    :param config: Experiment config.

    :returns: Depending on the backend, returns a module that is either a PyTorch distributed
        module, a fairscale fully sharded module or a deepspeed engine.
    """
    # Instantiate model on executor in case its too large for pickle and sent as a class.
    _sanitize_init_model_params(config)
    if config.backend == "ddp" and config.zero_lvl in [0, 1, 2]:
        module = MaggyDDPModuleWrapper.config(config.module)
    elif config.backend == "ddp":
        module = MaggyFairScaleModuleWrapper.config(config.module)
    elif config.backend == "deepspeed":
        module = MaggyDeepSpeedModuleWrapper.config(config.module, config.ds_config)
    return module


def _sanitize_init_model_params(config: DistributedConfig) -> None:
    assert inspect.isclass(
        config.module
    ), "Passed module should be a class, not an instance."
    if config.backend == "ddp":
        if config.ds_config:
            print(
                "Warning: DeepSpeed config passed for DDP backend. Config will be discarded."
            )
        if config.zero_lvl not in [0, 1, 2, 3]:
            raise ValueError(
                f"DeepSpeed level has to be in [0,1,2,3], is {config.zero_lvl}."
            )
        return
    if config.backend == "deepspeed":
        if not config.ds_config:
            raise ValueError(
                """DeepSpeed requires a configuration! For more information, see
                              https://www.deepspeed.ai/getting-started/#deepspeed-configuration"""
            )
        if config.ds_config and config.zero_lvl != 0:
            raise ValueError(
                "Zero level not supported with DeepSpeed, use ds_config instead!"
            )
        return
    raise ValueError(f"Unsupported backend {config.backend}.")


def _monkey_patch_pytorch(zero_lvl):
    # Patch DataLoader to always be distributed.
    torch.utils.data.DataLoader = MaggyDataLoader
    if zero_lvl > 0:
        torch.optim.Adam = MaggyZeroAdam
        torch.optim.SGD = MaggyZeroSGD
