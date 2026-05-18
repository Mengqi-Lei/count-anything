# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser
from copy import deepcopy

import submitit
import torch

from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers
from tqdm import tqdm


os.environ["HYDRA_FULL_ERROR"] = "1"


DEFAULT_SINGLE_STAGE_TRAINING_STAGE = "joint_counter_adaptation"
DEFAULT_PRETRAINED_SAM3_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "pretrained",
    "sam3.pt",
)
TRAINING_STAGE_ALIASES = {
    "stage1_p2p_head_only": "stage1_pdc_head_only",
    "p2p_branch_only": "pdc_branch_only",
    "p2p_only": "pdc_branch_only",
    "pdc_only": "pdc_branch_only",
    "stage2_joint_adaptation": "joint_counter_adaptation",
    "sam_det_only": "rsc_only",
    "sam_only": "rsc_only",
    "encoder_p2p_only": "encoder_pdc_only",
}


class SlurmEvent:
    QUEUED = "QUEUED"
    START = "START"
    FINISH = "FINISH"
    JOB_ERROR = "JOB_ERROR"
    SLURM_SIGNAL = "SLURM_SIGNAL"


def handle_custom_resolving(cfg):
    # We'll resolve the config here, so we can catch mistakes early.
    # However, we need to pass the un-resolved config to the launcher
    # (because DVC resolving needs to be done on the node it will run on)
    # First, do a copy without triggering resolving
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)
    return cfg_resolved


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    # assert cfg.launcher.gpus_per_node == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        # mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        job_env = submitit.JobEnvironment()
        # Need to add this again so the hydra.job.set_env PYTHONPATH
        # is also set when launching jobs.
        add_pythonpath_to_sys_path()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        register_omegaconf_resolvers()
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()

    def __call__(self):
        job_env = submitit.JobEnvironment()
        self.setup_job_info(job_env.job_id, job_env.global_rank)
        try:
            self.run_trainer()
        except Exception as e:
            # Log the exception. Then raise it again (as what SubmititRunner currently does).
            message = format_exception(e)
            logging.error(message)
            raise e

    def setup_job_info(self, job_id, rank):
        """Set up slurm job info"""
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,
        }

        self.has_setup = True


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def _resolve_cfg_value(cfg, key_path: str):
    probe = OmegaConf.create({"value": OmegaConf.select(cfg, key_path)})
    return OmegaConf.to_container(probe, resolve=True)["value"]


def _canonicalize_training_stage_name(training_stage):
    if training_stage is None:
        return None
    training_stage = str(training_stage)
    return TRAINING_STAGE_ALIASES.get(training_stage, training_stage)


def _apply_single_stage_defaults(cfg):
    trainer_mode = _resolve_cfg_value(cfg, "trainer.mode")
    if trainer_mode is not None and str(trainer_mode).lower() != "train":
        return

    trainer_stage = OmegaConf.select(cfg, "trainer.training_stage")
    scratch_stage = OmegaConf.select(cfg, "scratch.training_stage")
    training_stage = trainer_stage if trainer_stage is not None else scratch_stage
    canonical_training_stage = _canonicalize_training_stage_name(training_stage)

    if training_stage is None:
        if "scratch" not in cfg or cfg.scratch is None:
            cfg.scratch = OmegaConf.create({})
        cfg.scratch.training_stage = DEFAULT_SINGLE_STAGE_TRAINING_STAGE
        training_stage = DEFAULT_SINGLE_STAGE_TRAINING_STAGE
        canonical_training_stage = DEFAULT_SINGLE_STAGE_TRAINING_STAGE

    if "trainer" in cfg and OmegaConf.select(cfg, "trainer.training_stage") is None:
        cfg.trainer.training_stage = training_stage

    if canonical_training_stage == DEFAULT_SINGLE_STAGE_TRAINING_STAGE:
        if "paths" not in cfg or cfg.paths is None:
            cfg.paths = OmegaConf.create({})
        stage1_checkpoint_path = OmegaConf.select(cfg, "paths.stage1_checkpoint_path")
        if not stage1_checkpoint_path:
            cfg.paths.stage1_checkpoint_path = DEFAULT_PRETRAINED_SAM3_PATH


def _validate_stage2_checkpoint_loading(cfg):
    trainer_stage = OmegaConf.select(cfg, "trainer.training_stage")
    scratch_stage = OmegaConf.select(cfg, "scratch.training_stage")
    training_stage = trainer_stage if trainer_stage is not None else scratch_stage
    training_stage = _canonicalize_training_stage_name(training_stage)
    if training_stage not in {"joint_counter_adaptation", "rsc_only"}:
        return

    stage1_checkpoint_path = _resolve_cfg_value(cfg, "paths.stage1_checkpoint_path")
    if not stage1_checkpoint_path:
        raise ValueError(
            f"training_stage={training_stage} requires paths.stage1_checkpoint_path."
        )

    initializer_conf = OmegaConf.select(cfg, "trainer.checkpoint.model_weight_initializer")
    # Allow single-stage joint runs to start directly from the raw pretrained SAM3
    # weights file. In this case, model_builder loads `sam3.pt` and we should not
    # require the stage2-specific model-only initializer that expects a full stage1
    # training checkpoint.
    if (
        training_stage in {"joint_counter_adaptation", "rsc_only"}
        and initializer_conf is None
        and os.path.isfile(stage1_checkpoint_path)
        and os.path.basename(stage1_checkpoint_path) == "sam3.pt"
    ):
        return

    if initializer_conf is None:
        raise ValueError(
            f"training_stage={training_stage} requires "
            "trainer.checkpoint.model_weight_initializer to be configured."
        )

    initializer_checkpoint_path = _resolve_cfg_value(
        cfg, "trainer.checkpoint.model_weight_initializer.checkpoint_path"
    )
    if initializer_checkpoint_path != stage1_checkpoint_path:
        raise ValueError(
            f"training_stage={training_stage} requires stage2 initializer checkpoint "
            f"to match paths.stage1_checkpoint_path, but got "
            f"{initializer_checkpoint_path!r} vs {stage1_checkpoint_path!r}."
        )

    if not os.path.isfile(stage1_checkpoint_path):
        raise FileNotFoundError(
            f"training_stage={training_stage} expected stage1 checkpoint at "
            f"{stage1_checkpoint_path}, but it does not exist."
        )


def main(args) -> None:
    cfg = compose(config_name=args.config)
    experiment_log_dir_override = os.environ.get("SAM3_EXPERIMENT_LOG_DIR_OVERRIDE")
    if experiment_log_dir_override:
        cfg.launcher.experiment_log_dir = experiment_log_dir_override
        if "paths" in cfg and "experiment_log_dir" in cfg.paths:
            cfg.paths.experiment_log_dir = experiment_log_dir_override

    stage1_checkpoint_override = os.environ.get("SAM3_STAGE1_CHECKPOINT_PATH_OVERRIDE")
    if stage1_checkpoint_override and "paths" in cfg and "stage1_checkpoint_path" in cfg.paths:
        cfg.paths.stage1_checkpoint_path = stage1_checkpoint_override

    _apply_single_stage_defaults(cfg)

    if cfg.launcher.experiment_log_dir is None:
        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        cfg.launcher.experiment_log_dir = os.path.join(
            "outputs",
            config_stem,
        )
    _validate_stage2_checkpoint_loading(cfg)
    # 训练启动时只保留必要摘要，完整配置仍然写入 log_dir 下的 yaml 文件。
    print(f"Launching train config: {args.config}")

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    experiment_log_dir = cfg.launcher.experiment_log_dir
    print(f"Experiment Log Dir: {experiment_log_dir}")
    submitit_dir = os.path.join(experiment_log_dir, "submitit_logs")

    # Prioritize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    if submitit_conf.use_cluster:
        executor = submitit.AutoExecutor(folder=submitit_dir)
        submitit_conf.partition = (
            args.partition
            if args.partition is not None
            else submitit_conf.get("partition", None)
        )
        submitit_conf.account = (
            args.account
            if args.account is not None
            else submitit_conf.get("account", None)
        )
        submitit_conf.qos = (
            args.qos if args.qos is not None else submitit_conf.get("qos", None)
        )
        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,
            "name": (
                submitit_conf.name if hasattr(submitit_conf, "name") else args.config
            ),
            "slurm_partition": submitit_conf.partition,
            "gpus_per_node": cfg.launcher.gpus_per_node,
            "tasks_per_node": cfg.launcher.gpus_per_node,  # one task per GPU
            "cpus_per_task": submitit_conf.cpus_per_task,
            "nodes": cfg.launcher.num_nodes,
            "slurm_additional_parameters": {
                "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
            },
        }
        if "include_nodes" in submitit_conf:
            assert (
                len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes
            ), "Not enough nodes"
            job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
                submitit_conf["include_nodes"]
            )
        if submitit_conf.account is not None:
            job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
        if submitit_conf.qos is not None:
            job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos

        if submitit_conf.get("mem_gb", None) is not None:
            job_kwargs["mem_gb"] = submitit_conf.mem_gb
        elif submitit_conf.get("mem", None) is not None:
            job_kwargs["slurm_mem"] = submitit_conf.mem

        if submitit_conf.get("constraints", None) is not None:
            job_kwargs["slurm_constraint"] = submitit_conf.constraints

        if submitit_conf.get("comment", None) is not None:
            job_kwargs["slurm_comment"] = submitit_conf.comment

        # Supports only cpu-bind option within srun_args. New options can be added here
        if submitit_conf.get("srun_args", None) is not None:
            job_kwargs["slurm_srun_args"] = []
            if submitit_conf.srun_args.get("cpu_bind", None) is not None:
                job_kwargs["slurm_srun_args"].extend(
                    ["--cpu-bind", submitit_conf.srun_args.cpu_bind]
                )

        print("###################### SLURM Config ####################")
        print(job_kwargs)
        print("##########################################")
        executor.update_parameters(**job_kwargs)

        if (
            "job_array" in submitit_conf
            and submitit_conf.job_array.get("num_tasks", -1) > 0
        ):
            num_tasks = submitit_conf.job_array.num_tasks
            job_array_config_dir = os.path.join(
                cfg.launcher.experiment_log_dir, "job_array_configs"
            )
            makedir(job_array_config_dir)

            job_indices = range(num_tasks)
            ports = random.sample(
                range(submitit_conf.port_range[0], submitit_conf.port_range[1] + 1),
                k=len(job_indices),
            )

            jobs_runners_configs = []
            with executor.batch():
                task_index = 0
                for indices, main_port in tqdm(zip(job_indices, ports)):
                    curr_cfg = deepcopy(cfg)
                    curr_cfg.submitit.job_array["task_index"] = task_index
                    curr_cfg_resolved = handle_custom_resolving(cfg)
                    runner = SubmititRunner(main_port, curr_cfg)
                    job = executor.submit(runner)
                    jobs_runners_configs.append(
                        (job, runner, curr_cfg, curr_cfg_resolved)
                    )
                    task_index += 1

            for job, runner, job_cfg, job_cfg_resolved in jobs_runners_configs:
                print("Submitit Job ID:", job.job_id)

                # Save job specific config
                job_array_config_file = os.path.join(
                    job_array_config_dir, "{}.config.yaml".format(job.job_id)
                )
                with g_pathmgr.open(job_array_config_file, "w") as f:
                    f.write(OmegaConf.to_yaml(job_cfg))

                job_array_config_resolved_file = os.path.join(
                    job_array_config_dir, "{}.config_resolved.yaml".format(job.job_id)
                )
                with g_pathmgr.open(job_array_config_resolved_file, "w") as f:
                    f.write(OmegaConf.to_yaml(job_cfg_resolved, resolve=True))

                runner.setup_job_info(job.job_id, rank=0)
                # runner.log_event(event_type=SlurmEvent.QUEUED)
        else:
            main_port = random.randint(
                submitit_conf.port_range[0], submitit_conf.port_range[1]
            )
            runner = SubmititRunner(main_port, cfg)
            job = executor.submit(runner)
            print(f"Submitit Job ID: {job.job_id}")
            runner.setup_job_info(job.job_id, rank=0)

    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/roboflow_v100_full_ft_100_images.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    if os.path.isfile(args.config):
        config_path = os.path.abspath(args.config)
        initialize_config_dir(config_dir=os.path.dirname(config_path), version_base="1.2")
        args.config = os.path.splitext(os.path.basename(config_path))[0]
    else:
        initialize_config_module("sam3.train", version_base="1.2")
    register_omegaconf_resolvers()
    main(args)
