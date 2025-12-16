import gc, os, datetime

from omegaconf import OmegaConf
from easydict import EasyDict as edict
from hydra import initialize
from hydra import compose as hydra_compose
import argparse

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch

torch.multiprocessing.set_sharing_strategy("file_system")

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torchmetrics import AUROC

from roi_lift.util.opt import set_optimizer
from roi_lift.util.sch import set_scheduler
from roi_lift.util.system import set_seed
from roi_lift.util.metric import SliceDistAvg, SliceDistOver, SliceDistUnder
from roi_lift.data.embed_loader import EmbedDataset, EmbedLoader
from roi_lift.data.duke_loader import DukeDataset, DukeLoader
from roi_lift.model.registry import model_registry


def parse_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment/roi-lift-no-tv", type=str)
    args, _ = parser.parse_known_args()

    with initialize(config_path="config"):
        cfg = hydra_compose(config_name=args.experiment)
        _ = OmegaConf.set_struct(cfg, False)

    return edict(merge(dict(cfg), dict(vars(args))))


class Trainer(pl.LightningModule):

    def __init__(self, cfg):
        super(Trainer, self).__init__()

        self.output_d_d = {"embed": None, "duke": None}

        self.cfg = cfg
        self.save_hyperparameters(logger=True)

        print(f"Preparing model: {self.cfg.model.name} ...")
        self.net = model_registry[self.cfg.model.name](**self.cfg.model)

        self.metric_d = self.prepare_metrics()

        torch.cuda.empty_cache()
        gc.collect()

    def on_train_start(self):
        self.logger.log_hyperparams(self.cfg)
        try:
            self.net.set_tau_schedule(self.cfg.epoch)
            self.net.update_tau_after_epoch()
        except:
            pass

    def training_step(self, input_d, _):
        loss, _ = self.common_step("train", "embed", input_d)
        return loss

    def validation_step(self, input_d, _, dataloader_idx=0):

        dataset_name = {0: "embed", 1: "duke"}[dataloader_idx]

        _, output_d = self.common_step("valid", dataset_name, input_d)

        if "cam" not in self.cfg.model.name.lower():
            soft_hot = output_d["soft_hot"].to(torch.float32).detach().cpu()
        else:
            output_d_es = self.net.estimate_slice(input_d)
            output_d_fw = self.net(input_d)
            output_d = merge(output_d_es, output_d_fw)
            soft_hot = output_d["soft_hot"].to(torch.float32).detach().cpu()

        self.output_d_d[dataset_name] = output_d  # for visualization at epoch end

        gt_center = input_d["slice_center"].to(torch.float32).detach().cpu()
        for f in ["under", "over", "avg"]:
            self.metric_d["valid"][f"{dataset_name}_dist_{f}"].update(soft_hot, gt_center)

    def common_step(self, mode, dataset_name, input_d):

        output_d = self.net(input_d)

        if dataset_name == "duke":
            return None, output_d

        elif dataset_name == "embed":

            loss_d = self.net.compute_loss(output_d)

            for loss_name, loss_value in loss_d.items():
                self.my_log(f"{mode}/{loss_name}", loss_value)

            for pred_type in ["subtype", "severity"]:
                metric = self.metric_d[mode][pred_type]
                metric.update(
                    output_d[pred_type + "_logit"].as_tensor(),
                    output_d[pred_type].to(torch.long),
                )
            return loss_d["total_loss"], output_d

        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()
        gc.collect()

    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_epoch_end(self):
        self.common_epoch_end("train")

    def on_validation_epoch_end(self):

        self.common_epoch_end("valid")

        for dataset_name in ["duke", "embed"]:
            vis_output_d = self.net.vis(self.output_d_d[dataset_name])
            for vis_key, vis_val in vis_output_d.items():
                self.logger.experiment.add_image(
                    f"valid/{dataset_name}_{vis_key}",
                    vis_val.permute(-1, 0, 1),
                    self.current_epoch,
                )

        vis_output_d = self.net.vis(self.output_d_d["duke"])
        for vis_key, vis_val in vis_output_d.items():
            self.logger.experiment.add_image(f"valid/duke_{vis_key}", vis_val.permute(-1, 0, 1), self.current_epoch)

        for dataset_name in ["duke", "embed"]:
            for metric_type in ["under", "over", "avg"]:
                self.log(
                    f"valid/{dataset_name}_dist_{metric_type}",
                    self.metric_d["valid"][f"{dataset_name}_dist_{metric_type}"].compute(),
                    on_step=False,
                )

    def common_epoch_end(self, mode):

        for pred_type in ["subtype", "severity"]:
            metric = self.metric_d[mode][pred_type]
            self.log(f"{mode}/{pred_type}/auc", metric.compute(), on_step=False)
            metric.reset()

        # update tau if the model is applicable
        try:
            self.my_log(f"train/tau", self.net.tau, on_step=False)
            self.net.update_tau_after_epoch()
        except AttributeError:
            pass

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        model_optimizer = set_optimizer(self.net, opt_kwargs=self.cfg.optimizer)
        model_scheduler = set_scheduler(model_optimizer, sch_kwargs=self.cfg.scheduler)
        return [model_optimizer], [model_scheduler]

    def my_log(self, name, val, on_step=True):
        return self.log(name, val, on_step=on_step, on_epoch=True, logger=True, prog_bar=True)

    def prepare_metrics(self):

        def gen_auroc_metric(num_classes, task="multilabel"):
            if num_classes == 1:
                task = "binary"
            return AUROC(task=task, num_labels=num_classes, ignore_index=-1).to(self.device)

        metric_d = {}
        for mode in ["train", "valid", "test"]:
            metric_d[mode] = {}
            for prediction_type in ["subtype", "severity"]:
                metric = gen_auroc_metric(self.net.num_class_d[prediction_type])
                metric_d[mode][prediction_type] = metric

        for mode in ["valid", "test"]:
            for data_source in ["embed", "duke"]:
                metric_d[mode][f"{data_source}_dist_under"] = SliceDistUnder(**self.cfg.metric)
                metric_d[mode][f"{data_source}_dist_over"] = SliceDistOver(**self.cfg.metric)
                metric_d[mode][f"{data_source}_dist_avg"] = SliceDistAvg(**self.cfg.metric)

        return metric_d


def main():

    cfg = parse_arg()

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    current_datetime = f"{current_date}/{current_time}"
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    set_seed(cfg.seed_number)

    # Load datasets
    ###################################################
    train_dataset = EmbedDataset(
        **cfg.embed_dataset,
        transform_args=cfg.transform_train_args,
        mode="train",
    )
    train_loader = EmbedLoader(
        dataset=train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    valid_embed_dataset = EmbedDataset(
        **cfg.embed_dataset,
        transform_args=cfg.transform_valid_args,
        mode="valid",
    )
    valid_embed_loader = EmbedLoader(
        dataset=valid_embed_dataset,
        num_workers=cfg.num_workers,
        batch_size=1,
        shuffle=True,  # for random visualization
    )

    # validation does not limit
    valid_duke_dataset = DukeDataset(
        **cfg.duke_dataset,
        transform_args=cfg.transform_valid_args,
    )
    valid_duke_loader = DukeLoader(
        dataset=valid_duke_dataset,
        num_workers=cfg.num_workers,
        batch_size=1,
        shuffle=True,  # for random visualization
    )

    # callbacks
    ###################################################
    log_file_name = cfg.experiment_name
    log_save_dir = f"{current_file_dir}/{cfg.log_base_path}/{current_datetime}/"
    logger = TensorBoardLogger(log_save_dir, log_file_name, 0, default_hp_metric=True)

    ckpt_callback = ModelCheckpoint(
        dirpath=f"{current_file_dir}/{cfg.ckpt_base_path}/{current_datetime}/",
        filename=cfg.experiment_name,
        monitor="valid/embed_dist_avg",
        save_top_k=1,
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor, ckpt_callback]

    # init model and trainer
    ###################################################
    model = Trainer(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=cfg.precision,
        max_epochs=cfg.epoch,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        num_sanity_val_steps=10,
        gradient_clip_val=0.5,
        limit_train_batches=cfg.train_iteration_per_epoch,
        devices=cfg.num_gpu,
        strategy=(DDPStrategy(find_unused_parameters=True, static_graph=True) if cfg.num_gpu > 1 else "auto"),
        use_distributed_sampler=True if cfg.num_gpu > 1 else False,
    )

    trainer.fit(model, train_loader, [valid_embed_loader, valid_duke_loader])


if __name__ == "__main__":
    main()
