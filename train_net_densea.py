#!/usr/bin/env python
"""
DiffusionDet Training Script.
"""

import os
import copy
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import json
from PIL import ImageFilter
import cv2

import torch
import numpy as np
import time

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from fvcore.nn.precise_bn import get_bn_modules

from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances, BoxMode
from detectron2.engine import (
    DefaultTrainer, default_argument_parser, default_setup, launch,
    create_ddp_model, AMPTrainer, SimpleTrainer, hooks
)
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results, DatasetEvaluators
from detectron2.solver.build import maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.modeling import build_model
from torch.amp import autocast as cuda_autocast, GradScaler

from diffusiondet import (
    add_diffusiondet_config,
    DiffusionDetWithTTA
)

from diffusiondet import eval_history
from diffusiondet.fast_opt import flag as fast_flag, group_params, make_fast_optimizer_class
from diffusiondet.util.model_ema import (
    add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer,
    EMAHook, apply_model_ema_and_restore, EMADetectionCheckpointer
)

logger = logging.getLogger("detectron2")
logger.setLevel(logging.INFO)
class _GaussianBlurTransform(T.Transform):
    def __init__(self, radius: float): super().__init__(); self.radius = radius
    def apply_image(self, img: np.ndarray) -> np.ndarray: return cv2.GaussianBlur(img, ksize=(0,0), sigmaX=self.radius, sigmaY=self.radius)
    def apply_coords(self, coords: np.ndarray) -> np.ndarray: return coords

class RandomGaussianBlur(T.Augmentation):
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0): super().__init__(); self._init(locals())
    def get_transform(self, image: np.ndarray) -> T.Transform:
        if torch.rand(1) < self.p:
            radius = torch.empty(1).uniform_(self.radius_min, self.radius_max).item()
            return _GaussianBlurTransform(radius)
        return T.NoOpTransform()

class _NoiseTransform(T.Transform):
    def __init__(self, std_dev_abs: float): super().__init__(); self.std_dev_abs = std_dev_abs
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        noise = np.random.normal(0, self.std_dev_abs, img_float.shape)
        return np.clip(img_float + noise, 0, 255).astype(np.uint8)
    def apply_coords(self, coords: np.ndarray) -> np.ndarray: return coords

class RandomNoise(T.Augmentation):
    def __init__(self, p: float = 0.5, noise_std_dev_max: float = 0.03): super().__init__(); self._init(locals())
    def get_transform(self, image: np.ndarray) -> T.Transform:
        if torch.rand(1) < self.p:
            std_dev_abs = torch.empty(1).uniform_(0, self.noise_std_dev_max).item() * 255.0
            return _NoiseTransform(std_dev_abs)
        return T.NoOpTransform()
    
# ---------------------------------------------------------------------------- #
# 1) CUSTOM DATASET MAPPER FOR DATA AUGMENTATION
# ---------------------------------------------------------------------------- #
class MarineDebrisDatasetMapper:
    """
    Custom mapper that injects extra data augmentation
    while remaining compatible with object detection.
    """

    def __init__(self, cfg, is_train=True):
        """
        Args:
            cfg: detectron2 configuration
            is_train (bool): True for training
        """
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        # Resize according to cfg.INPUT (otherwise images would be used at native resolution
        # and MIN/MAX_SIZE in the yaml would have no effect).
        # It goes first so that the remaining augmentations work on the already resized image.
        if self.is_train:
            augmentations = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                                  cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]
        else:
            augmentations = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")]
        # Define several transforms
        if self.is_train:
            augmentations.extend([
                T.RandomFlip(horizontal=True, vertical=False), T.RandomBrightness(0.6, 1.4), 
                T.RandomContrast(0.6, 1.4), T.RandomSaturation(0.6, 1.4),
                T.RandomRotation(angle=[-20, 20], expand=False, sample_style="range"),
                RandomGaussianBlur(p=0.4, radius_min=0.5, radius_max=2.5), 
                RandomNoise(p=0.3, noise_std_dev_max=0.03), 
            ])
        self.tfm_gens = T.AugmentationList(augmentations)
        logger.info(f"MarineDebrisDatasetMapper (is_train={is_train}) using augmentations: {self.tfm_gens}")

    def __call__(self, dataset_dict):
        """
        dataset_dict: dict with:
          - "file_name": image path
          - "annotations": list of annotations (optional, if is_train)
          - ... (other metadata)
        Returns: modified dict with:
          - "image": tensor [C, H, W]
          - "instances": Instances object with gt_boxes, gt_classes...
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # Avoid mutating the original

        # 1. Load the image
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # 2. Apply the augmentation transforms
        aug_input = T.AugInput(image)
        transforms_applied = self.tfm_gens(aug_input)
        image_transformed = aug_input.image
        dataset_dict["image_shape"] = image_transformed.shape

        if "annotations" in dataset_dict:
            annos = []
            for obj in dataset_dict.get("annotations", []):
                if obj.get("iscrowd", 0) == 1: continue
                if "bbox" not in obj: continue
                bbox_mode = obj.get("bbox_mode", BoxMode.XYWH_ABS)
                bbox_xyxy = BoxMode.convert(obj["bbox"], bbox_mode, BoxMode.XYXY_ABS)
                instance_dict_to_transform = {"bbox": bbox_xyxy, "bbox_mode": BoxMode.XYXY_ABS}
                if "segmentation" in obj: instance_dict_to_transform["segmentation"] = obj["segmentation"]
                transformed_instance = utils.transform_instance_annotations(instance_dict_to_transform, transforms_applied, image_transformed.shape[:2])
                final_bbox = transformed_instance["bbox"]
                if final_bbox[2] > final_bbox[0] and final_bbox[3] > final_bbox[1]:
                    obj["bbox"] = final_bbox; obj["bbox_mode"] = transformed_instance["bbox_mode"]
                    if "segmentation" in transformed_instance: obj["segmentation"] = transformed_instance["segmentation"]
                    annos.append(obj)
            dataset_dict["annotations"] = annos
            instances = utils.annotations_to_instances(annos, image_transformed.shape[:2])
            if self.is_train: dataset_dict["instances"] = utils.filter_empty_instances(instances)
            else: dataset_dict["instances"] = instances
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_transformed.transpose(2, 0, 1)))
        return dataset_dict

# ---------------------------------------------------------------------------- #
# 2) SPECIALIZED TRAINER
# ---------------------------------------------------------------------------- #
class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        logger_trainer = logging.getLogger("detectron2")
        if not logger_trainer.isEnabledFor(logging.INFO): setup_logger()
        
        cfg_init = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        # GPU performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = self.build_model(cfg_init)

        # Convert to channels_last memory format for better GPU utilization
        self.model = self.model.to(memory_format=torch.channels_last)

        self.optimizer = self.build_optimizer(cfg_init, self.model)
        self.data_loader = self.build_train_loader(cfg_init)
        self._data_loader_iter = iter(self.data_loader)

        self.model = create_ddp_model(self.model, broadcast_buffers=False,
                                      find_unused_parameters=cfg_init.MODEL.BACKBONE.FREEZE_AT > -1 or cfg_init.get("FIND_UNUSED_PARAMETERS", False))
        self.accumulation_steps = cfg_init.SOLVER.GRAD_ACCUM_STEPS
        if self.accumulation_steps > 1:
            logger_trainer.info(f"Gradient accumulation enabled: {self.accumulation_steps} steps.")
            self.optimizer.zero_grad() 

        # AMP setup
        self.use_amp = cfg.SOLVER.AMP.ENABLED
        if self.use_amp:
            prec = cfg.SOLVER.AMP.PRECISION.lower()
            if prec == "fp16": self.amp_dtype = torch.float16
            elif prec == "bf16": self.amp_dtype = torch.bfloat16
            else: raise ValueError(f"Unsupported AMP precision: {cfg.SOLVER.AMP.PRECISION}")
            init_scale = getattr(cfg.SOLVER.AMP, "INIT_SCALE", None)
            self.grad_scaler = GradScaler(init_scale=init_scale) if init_scale else GradScaler()
            if fast_flag("DENSEA_NO_SCALER"):
                # bf16 has the same range as fp32: loss scaling (a power of 2, exact) brings nothing.
                # The protection is kept: a step with a non-finite gradient is skipped (see _step_without_scaler).
                if self.amp_dtype != torch.bfloat16:
                    raise ValueError("DENSEA_NO_SCALER=1 is only valid with bf16 AMP")
                self.grad_scaler = None
        else:
            self.grad_scaler = None
        self._fast_loss = fast_flag("DENSEA_FAST_LOSS")
        
        self.scheduler = Trainer.build_lr_scheduler(cfg_init, self.optimizer)

        kwargs_ema_checkpointer = {'trainer': weakref.proxy(self)}
        kwargs_ema_checkpointer.update(may_get_ema_checkpointer(cfg_init, self.model))

        self.checkpointer = DetectionCheckpointer(
            self.model, cfg_init.OUTPUT_DIR,
            optimizer=self.optimizer, scheduler=self.scheduler,
            **kwargs_ema_checkpointer)
        
        self.start_iter = 0 
        self.max_iter = cfg_init.SOLVER.MAX_ITER
        self.cfg = cfg_init
        
        # super().__init__ already registered a set of hooks (LRScheduler, EMAHook,
        # EvalHook, PeriodicCheckpointer...) bound to the first model. If they are not cleared, the LR
        # advances two steps per iteration (full cosine at double speed) and evaluation runs twice.
        self._hooks = []
        self.register_hooks(self.build_hooks())
        self._save_config()

    def _save_config(self): # Identical
        config_path = os.path.join(self.cfg.OUTPUT_DIR, 'config_runtime.yaml')
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        with open(config_path, 'w') as f: f.write(self.cfg.dump())
        logging.getLogger(__name__).info(f"Runtime configuration saved to {config_path}")

    def run_step(self): 
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        try: data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        
        # Forward with autocast
        with cuda_autocast(enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"):
            loss_dict = self.model(data) 
            losses = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
        
        if not self._fast_loss:
            self._check_loss(losses, loss_dict, "BEFORE backward")

        # Backward and optimization
        if self.grad_scaler is not None:
            self.grad_scaler.scale(losses).backward()
        else:
            losses.backward()

        # Gradient accumulation & step
        if (self.iter + 1) % self.accumulation_steps == 0:
            if self.use_amp and self.grad_scaler is None:
                self._step_without_scaler(losses, loss_dict)
            elif self.use_amp:
                if self._fast_loss:
                    self._check_loss(losses, loss_dict, "before the optimizer step")
                self.grad_scaler.unscale_(self.optimizer)
                if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE,
                        norm_type=self.cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE
                    )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                if self._fast_loss:
                    self._check_loss(losses, loss_dict, "before the optimizer step")
                if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE,
                        norm_type=self.cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE
                    )
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Logging
        storage = get_event_storage()
        if self._fast_loss:
            # a single GPU->CPU transfer instead of one per loss term
            keys = list(loss_dict.keys()) if isinstance(loss_dict, dict) else []
            vals = [loss_dict[k] for k in keys] + [losses]
            host = torch.stack([v.detach().float().reshape(()) for v in vals]).tolist()
            storage.put_scalars(**dict(zip(keys, host)))
            if "total_loss" not in keys:
                storage.put_scalar("total_loss", host[-1])
        elif isinstance(loss_dict, dict):
            metrics_to_log = {k: v.detach().cpu().item() for k,v in loss_dict.items()}
            storage.put_scalars(**metrics_to_log)
            if "total_loss" not in metrics_to_log :
                 storage.put_scalar("total_loss", losses.detach().cpu().item())
        else:
            storage.put_scalar("total_loss", losses.detach().cpu().item())
        storage.put_scalar("data_time", data_time)
        storage.put_scalar('time', time.perf_counter() - start)

    def _check_loss(self, losses, loss_dict, when):
        if torch.isnan(losses) or torch.isinf(losses):
            logger.error(f"NaN/Inf loss detected at iter {self.iter} {when}: {loss_dict}")
            raise FloatingPointError(f"Loss became NaN/Inf at iter {self.iter} {when}. loss_dict: {loss_dict}")

    def _step_without_scaler(self, losses, loss_dict):
        '''bf16 AMP step without GradScaler (DENSEA_NO_SCALER=1).
        A single synchronization: checks at once that the loss and the global gradient norm are finite.
        If the gradient is not finite the step is skipped, as GradScaler.step does.'''
        clip = self.cfg.SOLVER.CLIP_GRADIENTS
        params = [p for p in self.model.parameters() if p.grad is not None]
        if clip.ENABLED:
            total = torch.nn.utils.clip_grad_norm_(params, max_norm=clip.CLIP_VALUE, norm_type=clip.NORM_TYPE)
        else:
            total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in params])))
        loss_ok, grad_ok = torch.stack([torch.isfinite(losses), torch.isfinite(total)]).tolist()
        if not loss_ok:
            self._check_loss(losses, loss_dict, "before the optimizer step")
        if grad_ok:
            self.optimizer.step()
        else:
            logger.warning(f"Non-finite gradient at iter {self.iter}: optimizer step skipped")

        
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # Set up EMA if enabled in the config
        if cfg.MODEL_EMA.ENABLED: may_build_model_ema(cfg, model)
        return model
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """Build the learning-rate scheduler from the config."""
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Return a COCO or LVIS evaluator according to dataset_name.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"No evaluator found for dataset {dataset_name} with type {evaluator_type}"
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Override this method to use our MarineDebrisDatasetMapper.
        """
        mapper = MarineDebrisDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Override to use our MarineDebrisDatasetMapper in test mode.
        On Windows with spawn multiprocessing, NUM_WORKERS > 0 can freeze
        during periodic evaluation, so num_workers=0 by default. On Linux
        (fork) it can be raised with DENSEA_TEST_WORKERS; InferenceSampler preserves the order,
        so the predictions do not change.
        """
        mapper = MarineDebrisDatasetMapper(cfg, is_train=False)
        workers = int(os.environ.get("DENSEA_TEST_WORKERS", "0"))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper, num_workers=workers)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build the optimizer (SGD or AdamW), with gradient clipping support.
        """
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        fast_opt = fast_flag("DENSEA_FAST_OPT")
        legacy_order = None
        if fast_opt:
            # one group per (lr, weight_decay) instead of one per tensor: same hyperparameters per parameter
            params, legacy_order = group_params(model, cfg.SOLVER.BASE_LR, cfg.SOLVER.WEIGHT_DECAY,
                                                cfg.SOLVER.BACKBONE_MULTIPLIER)
        for key, value in ([] if fast_opt else model.named_parameters(recurse=True)):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        def wrap(optim):
            optim = maybe_add_full_model_gradient_clipping(optim)
            return make_fast_optimizer_class(optim, cfg.SOLVER.CLIP_GRADIENTS) if fast_opt else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = wrap(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = wrap(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"No optimizer type {optimizer_type}")
        if fast_opt:
            optimizer.legacy_order = legacy_order
            logging.getLogger(__name__).info(
                f"DENSEA_FAST_OPT: {len(optimizer.param_groups)} parameter groups "
                f"({len(legacy_order)} tensors), foreach={optimizer.defaults.get('foreach')}")
        elif not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        """
        If EMA is enabled, evaluate with EMA; otherwise evaluate normally.
        """
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        """Run inference with Test-Time Augmentation."""
        logger.info("Running inference with test-time augmentation (TTA)...")
        # Build TTA-specific evaluators if needed (different output directories)
        evaluators = []
        for name in cfg.DATASETS.TEST:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_TTA", name)
            evaluator = cls.build_evaluator(cfg, name, output_folder=output_folder)
            evaluators.append(evaluator)
        evaluators = DatasetEvaluators(evaluators) # Combine if there are several

        # Wrap the model with TTA and run the evaluation (with or without EMA)
        tta_model = DiffusionDetWithTTA(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            res = cls.ema_test(cfg, tta_model, evaluators=evaluators)
        else:
            res = cls.test(cfg, tta_model, evaluators=evaluators)

        # Add the _TTA suffix to the result keys
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        List of hooks (checkpoint, eval, logger, etc.).
        """
        cfg = self.cfg.clone()
        cfg.defrost()

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,
            hooks.LRScheduler(),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        # Remove None from the list if EMA or PreciseBN were not added
        ret = [hook for hook in ret if hook is not None]

        if comm.is_main_process():
            if cfg.DENSEA.CHECKPOINT_KEEP_EVERY > 0:  # keep the periodic checkpoints
                ret.append(SparsePeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD,
                                                      cfg.DENSEA.CHECKPOINT_KEEP_EVERY))
            else:
                ret.append(hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=5 # Keep the last 5 checkpoints
                ))
        def test_and_save_results():
            # Evaluate with EMA if enabled
            self._last_eval_results = self.ema_test(self.cfg, self.model)

            # Save metrics to a JSON file (main process only)
            if comm.is_main_process():
                 metrics_path = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')
                 try:
                     with open(metrics_path, 'w') as f:
                         # Convert numpy types to native Python types for JSON
                         serializable_results = convert_dict_values_to_serializable(self._last_eval_results)
                         json.dump(serializable_results, f, indent=4)
                     if cfg.DENSEA.BEST_METRIC:  # history used to resume (see eval_history)
                         flat = {f"{t}/{m}": v for t, r in serializable_results.items()
                                 if isinstance(r, dict) for m, v in r.items()}
                         if cfg.DENSEA.BEST_METRIC in flat:
                             eval_history.append_eval(cfg.OUTPUT_DIR, self.iter,
                                                      cfg.DENSEA.BEST_METRIC,
                                                      flat[cfg.DENSEA.BEST_METRIC])
                     logger.info(f"Metrics saved to {metrics_path}")
                 except Exception as e:
                     logger.error(f"Failed to save metrics to {metrics_path}: {e}")

            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        # Best checkpoint and early stopping on the (EMA) AP of DATASETS.TEST
        if comm.is_main_process() and cfg.DENSEA.BEST_METRIC:
            ret.append(SeededBestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                              cfg.DENSEA.BEST_METRIC, mode="max", file_prefix="model_best",
                                              output_dir=cfg.OUTPUT_DIR))
        if cfg.DENSEA.EARLY_STOP_PATIENCE > 0:
            ret.append(EarlyStoppingHook(cfg.TEST.EVAL_PERIOD, cfg.DENSEA.BEST_METRIC,
                                         cfg.DENSEA.EARLY_STOP_PATIENCE, cfg.DENSEA.EARLY_STOP_MIN_DELTA,
                                         output_dir=cfg.OUTPUT_DIR))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

class SeededBestCheckpointer(hooks.BestCheckpointer):
    """BestCheckpointer that, on resume, starts from the previous best value (otherwise the first
    evaluation after resuming would overwrite model_best even if it were worse)."""

    def __init__(self, *args, output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_dir = output_dir

    def before_train(self):
        if self._output_dir and self.trainer.start_iter > 0:
            evals = eval_history.load_evals(self._output_dir, self._val_metric, self.trainer.start_iter)
            self.best_metric, self.best_iter = eval_history.replay_best(evals)
            logger.info(f"BestCheckpointer resumed: best={self.best_metric} @ {self.best_iter}")


class SparsePeriodicCheckpointer(hooks.PeriodicCheckpointer):
    """Saves every `period` iterations (for resuming) and, when saving the next one, deletes the
    previous one unless it falls on a multiple of `keep_every` (checkpoints that are kept)."""

    def __init__(self, checkpointer, period, keep_every):
        super().__init__(checkpointer, period, max_to_keep=None)
        self._keep_every, self._prev = keep_every, None

    def step(self, iteration, **kwargs):
        super().step(iteration, **kwargs)
        if (iteration + 1) % self.period != 0:
            return
        prev, self._prev = self._prev, iteration
        if prev is not None and (prev + 1) % self._keep_every != 0:
            path = os.path.join(self.checkpointer.save_dir, f"{self.file_prefix}_{prev:07d}.pth")
            if self.path_manager.exists(path):  # same as fvcore with max_to_keep
                self.path_manager.rm(path)


class EarlyStop(Exception):
    """Early-stopping signal (caught in main)."""


class EarlyStoppingHook(hooks.HookBase):
    """Stop training if `metric` does not improve by `min_delta` for `patience` evaluations."""

    def __init__(self, eval_period, metric, patience, min_delta, output_dir=None):
        self._period, self._metric = eval_period, metric
        self._patience, self._min_delta = patience, min_delta
        self._output_dir = output_dir
        self.best, self.bad = None, 0

    def before_train(self):
        # On resume, patience counts the whole run: it is rebuilt from the previous evaluations
        if self._output_dir and self.trainer.start_iter > 0:
            evals = eval_history.load_evals(self._output_dir, self._metric, self.trainer.start_iter)
            self.best, self.bad = eval_history.replay_patience(evals, self._min_delta)
            logger.info(f"EarlyStopping resumed: {len(evals)} previous evaluations, "
                        f"best={self.best}, no improvement={self.bad}/{self._patience}")

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period <= 0 or next_iter % self._period != 0:
            return
        val = self.trainer.storage.latest().get(self._metric)
        if val is None:
            return
        val = val[0]
        if self.best is None or val > self.best + self._min_delta:
            self.best, self.bad = val, 0
            return
        self.bad += 1
        logger.info(f"EarlyStopping: {self._metric}={val:.3f} no improvement ({self.bad}/{self._patience})")
        if self.bad >= self._patience:
            raise EarlyStop(f"early stopping at iteration {next_iter} (best {self.best:.3f})")


def add_densea_configs(cfg):
    """Extra config keys (the defaults keep the original behavior)."""
    from detectron2.config import CfgNode as CN
    cfg.DENSEA = CN()
    cfg.DENSEA.BEST_METRIC = ""          # e.g. "bbox/AP": saves model_best.pth
    cfg.DENSEA.EARLY_STOP_PATIENCE = 0   # number of evaluations without improvement; 0 = disabled
    cfg.DENSEA.EARLY_STOP_MIN_DELTA = 0.1  # in AP points (detectron2 uses a 0-100 scale)
    cfg.DENSEA.CHECKPOINT_KEEP_EVERY = 0   # iterations; >0 keeps those periodic checkpoints
                                           # (and only the latest of the others); 0 = last 5


def dataset_paths(data_root):
    """Annotation JSON files and image root; environment variables DENSEA_{TRAIN,VALID,TEST}_JSON and
    DENSEA_IMAGE_ROOT override the defaults."""
    root = os.environ.get("DENSEA_IMAGE_ROOT", data_root)
    return {name: (os.environ.get(f"DENSEA_{key}_JSON", os.path.join(data_root, rel)), root)
            for name, key, rel in (("densea_train", "TRAIN", "annotations/train.json"),
                                   ("densea_valid", "VALID", "annotations/val.json"),
                                   ("densea_test", "TEST", "annotations/test.json"))}


# Helper to convert numpy values to native types for JSON
def convert_dict_values_to_serializable(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = convert_dict_values_to_serializable(v)
        elif isinstance(v, (np.float32, np.float64)):
            new_dict[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            new_dict[k] = int(v)
        elif isinstance(v, list):
             # Convert the list elements if they are numpy types
             new_dict[k] = [float(i) if isinstance(i, (np.float32, np.float64)) else
                            int(i) if isinstance(i, (np.int32, np.int64)) else i
                            for i in v]
        else:
            new_dict[k] = v
    return new_dict

# ---------------------------------------------------------------------------- #
# 3) TRAINING SETUP AND MAIN
# ---------------------------------------------------------------------------- #
def setup(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_densea_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Create the output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, args)
    # Set up the main logger after creating OUTPUT_DIR
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron2")

    return cfg

def register_datasets(data_root):
    """Register all the required datasets in COCO format."""
    logger.info(f"Registering datasets from root: {data_root}")

    # --- Common metadata ---
    # Make sure the classes here match exactly those in your JSON files
    densea_metadata = {
        # "thing_classes": [
        #     "Basket", "Bottle", "Can", "Car_Bumper", "Fishing_Net",
        #     "Glove", "Mask", "Metal_Chain", "Metal_Debris", "Packaging_Bag",
        #     "Pipe", "Plastic_Bag", "Plastic_Debris", "Rope", "Shoe",
        #     "Squared_Can", "Tire", "Towel", "WashingMachine", "Wood"
        # ]
        "thing_classes": [
            "Bottle", "Can", "Fishing_Net",
            "Glove", "Mask", "Metal_Debris",
            "Plastic_Debris", "Tire"
        ]
    }
    num_classes = len(densea_metadata["thing_classes"])
    logger.info(f"Number of classes defined: {num_classes}")

    datasets_to_register = dataset_paths(data_root)

    for name, (json_file, image_root) in datasets_to_register.items():
        json_path = json_file
        image_path = image_root
        # Check that the files/directories exist before registering
        if os.path.exists(json_path) and os.path.isdir(image_path):
            logger.info(f"Registering: {name}")
            register_coco_instances(name, densea_metadata, json_path, image_path)
            # Set evaluator_type for each registered dataset
            MetadataCatalog.get(name).evaluator_type = "coco" # or "lvis" if applicable
            try:
                data = DatasetCatalog.get(name)
                logger.info(f"Dataset '{name}' registered with {len(data)} images.")
            except Exception as e:
                logger.error(f"Failed to verify the registration of '{name}': {e}")
        else:
            logger.warning(f"Could not register '{name}': JSON ('{json_path}') or image directory ('{image_path}') not found.")

    logger.info("Dataset registration completed.")
    return num_classes

def main(args):
    cfg = setup(args)

    # --- Register datasets ---
    # Root of the DenSea-Aggregated package (annotations/ and images/); override the individual
    # files with DENSEA_{TRAIN,VALID,TEST}_JSON and DENSEA_IMAGE_ROOT.
    data_root = os.environ.get("DENSEA_DATA_ROOT", ".")
    logger.info(f"Dataset root: {data_root}")
    num_classes = register_datasets(data_root)
    cfg.defrost()
    model_cfg_updated = False

    # --- Update the number of classes in the config ---
    # Generic example (adjust to your base config):
    if hasattr(cfg.MODEL, 'ROI_HEADS'): # For Faster R-CNN-style architectures
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        model_cfg_updated = True
    if hasattr(cfg.MODEL, 'DiffusionDet') and hasattr(cfg.MODEL.DiffusionDet, 'NUM_CLASSES'):
         cfg.MODEL.DiffusionDet.NUM_CLASSES = num_classes
         model_cfg_updated = True
    elif hasattr(cfg.MODEL, 'num_classes'): # Sometimes it is directly under MODEL
         cfg.MODEL.num_classes = num_classes
         model_cfg_updated = True
    
    if not model_cfg_updated: logger.warning("No standard NUM_CLASSES field found in cfg.MODEL.")
    else: logger.info(f"Config updated with NUM_CLASSES = {num_classes}")
    
    # Freeze the config after the final adjustments
    cfg.freeze()

    # Save the updated config again (optional, useful for debugging)
    if comm.is_main_process():
        config_path = os.path.join(cfg.OUTPUT_DIR, 'config_final_updated.yaml')
        with open(config_path, 'w') as f: f.write(cfg.dump())

    # Evaluation-only mode
    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Normal training
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    try:
        trainer.train()
    except EarlyStop as e:
        logger.info(f"Training finished: {e}")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )