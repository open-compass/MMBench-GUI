import json
import os
import subprocess

from vlmeval.smp import *
from models import LocalModelWrapper, APIModelWrapper
from benchmarks import BENCHMARK
from ipdb import set_trace as st
from utils.inference_tools import infer_data_local, infer_data_api
from utils.misc import load_env as load_env_file


# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if CUDA_VISIBLE_DEVICES != "":
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(",")]
        return gpu_list
    try:
        ps = subprocess.Popen(("nvidia-smi", "--list-gpus"), stdout=subprocess.PIPE)
        output = subprocess.check_output(("wc", "-l"), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


def init_env():
    global RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, LOCAL_RANK, GPU_LIST, CUDA_VISIBLE_DEVICES

    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 1))

    GPU_LIST = get_gpu_list()

    if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
        NGPU = len(GPU_LIST)
        assert (
            NGPU >= LOCAL_WORLD_SIZE
        ), "The number of processes should be less than or equal to the number of GPUs"
        GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
        DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
        CUDA_VISIBLE_DEVICES = [
            str(i) for i in GPU_LIST[DEVICE_START_IDX : DEVICE_START_IDX + GPU_PER_PROC]
        ]
        CUDA_VISIBLE_DEVICES = ",".join(CUDA_VISIBLE_DEVICES)
        # Set CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(
            f"RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},"
            f"LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}"
        )


def build_dataset_from_config(cfg, dataset_name):
    import inspect

    assert (
        dataset_name in BENCHMARK
    ), f"Dataset {dataset_name} is not supported in BENCHMARK. Please check your config file."

    cls = BENCHMARK[dataset_name]
    config = cp.deepcopy(cfg[dataset_name])
    if config == {} or (not isinstance(config, dict)):
        ValueError(
            f"Pealse check the config for dataset {dataset_name}, it should be a non-empty dictionary. "
        )

    sig = inspect.signature(cls.__init__)
    valid_params = {k: v for k, v in config.items() if k in sig.parameters}

    return cls(**valid_params)


def parse_args():
    help_msg = """\
You can launch the evaluation by setting --config.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
        {
            "model": {
                "uitars-1.5-7b-local": {
                    "model_path": "/mnt/petrelfs/wangxuehui/project/computer_use/MMBench-GUI/checkpoint/UI-TARS-1.5-7B",
                    "generate_cfg": {
                        "max_new_tokens": 512
                    },
                    "imp_type": "transformers",
                    "generate_function": "generate",
                    "preprocess_function": "models.local_uitars.preprocess_uitars",
                    "postprocess_function": "models.local_uitars.postprocess_uitars",
                    "custom_prompt": {
                        "GUIElementGrounding": "models.local_uitars.build_custom_prompt"
                    },
                    "kwargs": {
                        "system_prompt": "model_default",
                        "max_pixels": 2116800,
                        "min_pixels": 3136,
                        "img_size": -1,
                        "img_detail": "low"
                    }
                }
            },
            "data": {
                "GUIElementGrounding": {
                    "mode": "all",
                    "parse_function": "models.local_uitars.parse_grounding_response"
                },
                "GUIContentUnderstanding": {
                    "mode": "all",
                    "parse_function": "models.local_uitars.parse_understanding_response",
                    "match_mode": "exact_match"
                }
            }
        }
    ```
"""
    parser = argparse.ArgumentParser(
        description=help_msg, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=str, help="Path to the Config Json File")
    # Work Dir
    parser.add_argument(
        "--work-dir", type=str, default="./outputs", help="select the output directory"
    )
    # Infer + Eval or Infer Only
    parser.add_argument("--mode", type=str, default="all", choices=["all", "infer"])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument("--api-nproc", type=int, default=4, help="Parallel API calling")
    parser.add_argument(
        "--retry", type=int, default=None, help="retry numbers for API VLMs"
    )
    parser.add_argument(
        "--judge-args", type=str, default=None, help="Judge arguments in JSON format"
    )
    # Explicitly Set the Judge Model
    parser.add_argument("--judge", type=str, default=None)
    # Logging Utils
    parser.add_argument("--verbose", action="store_true")
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument("--ignore", action="store_true", help="Ignore failed indices. ")
    # Reuse: will reuse the existing prediction files
    parser.add_argument("--reuse", action="store_true")
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument(
        "--reuse-aux", type=bool, default=True, help="reuse auxiliary evaluation files"
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="use vllm to generate, the flag is only supported in Llama4 for now",
    )
    # Ignored variables
    parser.add_argument("--data", type=str, nargs="+", help="Names of Datasets")
    parser.add_argument("--model", type=str, nargs="+", help="Names of Models")

    args = parser.parse_args()
    return args


def main():
    logger = get_logger("MMBench-GUI")
    args = parse_args()
    use_config, cfg = False, None
    if args.config is not None:
        use_config, cfg = True, load(args.config)
        args.model = list(cfg["model"].keys())
        args.data = list(cfg["data"].keys())
    else:
        raise ValueError("You should set --config with a json file. ")

    if RANK == 0:
        if not args.reuse:
            logger.warning(
                "--reuse is not set, will not reuse previous (before one day) temporary files"
            )
        else:
            logger.warning(
                "--reuse is set, will reuse the latest prediction & temporary pickle files"
            )

    if "EVAL_WORK_DIR" in os.environ:
        args.work_dir = os.environ["EVAL_WORK_DIR"]

    if WORLD_SIZE > 1:
        import torch.distributed as dist

        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(
                seconds=int(os.environ.get("DIST_TIMEOUT", 3600))
            ),
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr("day"), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode="dir")
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        if use_config:
            if listinstr(
                ["http://", "https://"], cfg["model"][model_name]["model_path"]
            ):
                model = APIModelWrapper(
                    cfg["model"][model_name]["model_path"],
                    generate_cfg=cfg["model"][model_name]["generate_cfg"],
                    imp_type=cfg["model"][model_name]["imp_type"],
                    generate_function=cfg["model"][model_name].get(
                        "generate_function", None
                    ),
                    preprocess_function=cfg["model"][model_name].get(
                        "preprocess_function", None
                    ),
                    postprocess_function=cfg["model"][model_name].get(
                        "postprocess_function", None
                    ),
                    custom_prompt=cfg["model"][model_name].get("custom_prompt", None),
                    **cfg["model"][model_name]["kwargs"],
                )  # noqa: E501
            else:
                model = LocalModelWrapper(
                    cfg["model"][model_name]["model_path"],
                    generate_cfg=cfg["model"][model_name]["generate_cfg"],
                    imp_type=cfg["model"][model_name]["imp_type"],
                    generate_function=cfg["model"][model_name].get(
                        "generate_function", "generate"
                    ),
                    preprocess_function=cfg["model"][model_name].get(
                        "preprocess_function", "models.base.default_preprocess_function"
                    ),
                    postprocess_function=cfg["model"][model_name].get(
                        "postprocess_function",
                        "models.base.default_postprocess_function",
                    ),
                    custom_prompt=cfg["model"][model_name].get("custom_prompt", None),
                    **cfg["model"][model_name]["kwargs"],
                )  # noqa: E501
        else:
            ValueError("Currently, you should set --config. ")

        for _, dataset_name in enumerate(args.data):
            if WORLD_SIZE > 1:
                dist.barrier()

            try:
                result_file_base = f"{model_name}_{dataset_name}.xlsx"

                if use_config:
                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset_from_config(
                                cfg["data"], dataset_name
                            )
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg["data"], dataset_name)
                    if dataset is None:
                        logger.error(
                            f"Dataset {dataset_name} is not valid, will be skipped. "
                        )
                        continue
                else:
                    raise ValueError("You should set --config. ")

                result_file = osp.join(pred_root, result_file_base)

                # Reuse the previous prediction file if exists
                if RANK == 0 and len(prev_pred_roots):
                    prev_result_files = []
                    prev_pkl_file_list = []
                    for root in prev_pred_roots[::-1]:
                        if osp.exists(osp.join(root, result_file_base)):
                            if args.reuse_aux:
                                prev_result_files = fetch_aux_files(
                                    osp.join(root, result_file_base)
                                )
                            else:
                                prev_result_files = [osp.join(root, result_file_base)]
                            break
                        elif commit_id in root and len(ls(root)) and root != pred_root:
                            temp_files = ls(root, match=[dataset_name, ".pkl"])
                            if len(temp_files):
                                prev_pkl_file_list.extend(temp_files)
                                break
                    if not args.reuse:
                        prev_result_files = []
                        prev_pkl_file_list = []
                    if len(prev_result_files):
                        for prev_result_file in prev_result_files:
                            src = prev_result_file
                            tgt = osp.join(pred_root, osp.basename(src))
                            if not osp.exists(tgt):
                                shutil.copy(src, tgt)
                                logger.info(
                                    f"--reuse is set, will reuse the prediction file {src}."
                                )
                            else:
                                logger.warning(f"File already exists: {tgt}")

                    elif len(prev_pkl_file_list):
                        for fname in prev_pkl_file_list:
                            target_path = osp.join(pred_root, osp.basename(fname))
                            if not osp.exists(target_path):
                                shutil.copy(fname, target_path)
                                logger.info(
                                    f"--reuse is set, will reuse the prediction pickle file {fname}."
                                )
                            else:
                                logger.warning(f"File already exists: {target_path}")

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Perform the Inference
                if model.is_api:
                    model = infer_data_api(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        ignore_failed=args.ignore,
                    )
                else:
                    model = infer_data_local(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        ignore_failed=args.ignore,
                    )

                # Set the judge kwargs first before evaluation or dumping
                judge_kwargs = {
                    "nproc": args.api_nproc,
                    "verbose": args.verbose,
                    "retry": args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.retry is not None:
                    judge_kwargs["retry"] = args.retry
                if args.judge is not None:
                    judge_kwargs["model"] = args.judge
                else:
                    if dataset.TYPE in [
                        "GUI_Element_Grounding",
                        "GUI_Content_Understanding",
                    ]:  # Default to `exact matching` for GUI Element Grounding task and GUI Content Understanding task, which is more faster than running a judge model.
                        judge_kwargs["model"] = "exact_matching"
                    else:
                        judge_kwargs["model"] = "qwen-72b"

                if RANK == 0:
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Only RANK 0 handles the evaluation part
                if RANK == 0:
                    # Skip the evaluation part if only infer
                    if args.mode == "infer":
                        continue

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get("EVAL_PROXY", None)
                    old_proxy = os.environ.get("HTTP_PROXY", "")
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(
                            eval_results, pd.DataFrame
                        )
                        logger.info(
                            f"The evaluation of model {model_name} x dataset {dataset_name} has finished! "
                        )
                        logger.info("Evaluation Results:")
                        if isinstance(eval_results, dict):
                            logger.info("\n" + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info("\n" + tabulate(eval_results))

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    files = os.listdir(pred_root)
                    files = [
                        x
                        for x in files
                        if (f"{model_name}_{dataset_name}" in x or "status.json" in x)
                    ]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        if osp.exists(link_addr) or osp.islink(link_addr):
                            os.remove(link_addr)
                        os.symlink(file_addr, link_addr)

            except Exception as e:
                logger.exception(
                    f"Model {model_name} x Dataset {dataset_name} combination failed: {e}, "
                    "skipping this combination."
                )
                continue

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # st()
    init_env()
    load_env_file()
    main()
