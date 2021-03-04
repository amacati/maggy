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
from types import SimpleNamespace

import torch
import torch.distributed as dist

import numpy as np
import deepspeed
import fairscale

from maggy import util, tensorboard
from maggy.core.rpc import Client
from maggy.core.reporter import Reporter
from maggy.core.patching import MaggyDataLoader  # , MaggyTrainer
from maggy.core.environment.singleton import EnvSing

# import pytorch_lightning

# Patch DataLoader to always be distributed.
torch.utils.data.DataLoader = MaggyDataLoader

# Patch pytorch_lightning.Trainer to run with distributed config.
# pytorch_lightning.Trainer = MaggyTrainer


def dist_executor_fct(
    train_fn, config, app_id, run_id, server_addr, hb_interval, secret, log_dir,
):
    """
    Wraps the user supplied training function in order to be passed to the Spark Executors.
    Args:
        :param train_fn: Original training function.
        :type train_fn: callable
        :param config: Experiment config.
        :type config: DistributedConfig
        :param app_id: Maggy application ID.
        :type app_id: int
        :param run_id: Maggy run ID.
        :type run_id: int
        :param server_addr: IP of the Maggy worker registration RPC server.
        :type server_addr: str
        :param hb_interval: Worker heartbeat interval.
        :type hb_interval: Union[float, int]
        :param secret: Secret string to authenticate messages.
        :type secret: str
        :param log_dir: Location of the logger file directory on the file system.
        :type log_dir: str
    """

    def wrapper_function(_):
        """
        Patched function from prepare_function factory.
        Args:
            _ (object): Necessary sink for the iterator given by Spark to the function upon foreach
                calls. Can safely be disregarded.
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
                "NCCL_DEBUG": "INFO",
            }
            reporter.log(f"Torch config is {torch_config}")

            _setup_torch_env(torch_config)
            _init_cluster(timeout=60, random_seed=0)
            model = _init_model(config)

            reporter.log("Starting distributed training.", True)
            sig = inspect.signature(train_fn)
            if sig.parameters.get("reporter", None):
                retval = train_fn(
                    model=model,
                    train_set=config.train_set,
                    test_set=config.test_set,
                    reporter=reporter,
                )
            else:
                retval = train_fn(
                    model=model, train_set=config.train_set, test_set=config.test_set,
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


def _register_with_servers(client, reporter, partition_id):
    """Registers own address with server and starts heartbeat protocol.
    Args:
        client (Client): Client for communication with the server.
        reporter (Reporter): Reporter responsible for heartbeat.
        partition_id (int): Executors partition ID from Sparks RDD.
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


def _get_open_port():
    """Lets the OS choose a free socket and attempts to bind it.
    Returns:
        port (str): The port name.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to 0 lets OS choose a free socket.
    port = sock.getsockname()[1]
    sock.close()
    return port


def _setup_logging(reporter, log_dir):
    """Sets up logging directories and files, registers with tensorboard.
    Args:
        reporter (Reporter): Reporter responsible for logging.
        log_dir (str): Log directory path on the file system.
    Returns:
        (tuple): Tuple containing:
            tb_logdir (str): Path of the tensorboard directory.
            trial_log_file (str): Path of the trial log file.
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
    tensorboard._register(tb_logdir)
    return tb_logdir, trial_log_file


def _setup_torch_env(torch_config):
    """Registers the Torch config as environment variables on the worker.
    Args:
        torch_config (dict): Dictionary containing the values of the variables.
    """
    for env_variable in torch_config.keys():
        os.environ[env_variable] = str(torch_config[env_variable])


def _init_cluster(timeout=60, random_seed=0, backend="ddp"):
    """Checks if config is set, initializes the Torch distributed cluster and sets random seeds.

    Args:
        timeout (:obj:'int', optional): Time until initialization times out. Defaults to 60.
        random_seed (:obj:'int', optional): Random seed for Torch, numpy, random. Defaults to 0.
        backend (str): The backend that torch uses for distributed training. Either "ddp",
            "fairscale" or "deepspeed".

    Raises:
        KeyError: Checks on environment variables failed.
        RuntimeError: Checks on PyTorch distributed backend failed.
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


def _init_model(config):
    """Initializes the correct model according to `backend`.

    If the model is too large to be pickled, it's passed as a class instead and is instantiated
    here. The model gets further wrapped according to the backend's requirements.

    Args:
        config (DistributedConfig): Experiment config.

    Returns:
        model (torch.nn.Module): Depending on the backend, model is either a PyTorch distributed
            module, a fairscale module or a deepspeed engine.
    """
    # Instantiate model on executor in case its too large for pickle and sent as a class.
    if inspect.isclass(config.model):
        config.model = config.model()
    if config.backend == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(config.model.cuda())
    elif config.backend == "fairscale":
        model = fairscale.nn.FullyShardedDataParallel(config.model.cuda())
    elif config.backend == "deepspeed":
        ds_args = SimpleNamespace(local_rank=0)
        ds_config = {
            "train_micro_batch_size_per_gpu": 64,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.00015}},
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        }
        model = config.model
        model, *_ = deepspeed.initialize(
            args=ds_args,
            model=model,
            model_parameters=model.parameters(),
            config_params=ds_config,
        )
    return model
