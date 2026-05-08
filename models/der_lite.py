"""
DER-Lite v4: Multi-scale adapters + KD distillation loss.

Key design:
- Backbone frozen after task 0 (0 trainable backbone params per incremental task)
- Multi-scale adapters (~64K params each) operating on all 3 backbone fmaps
- Single-stage training (no FC reinit, no adapter pruning)
- KD distillation loss from old model (like iCaRL) to preserve old class knowledge
- Old adapters frozen to preserve old task features
- Per-task trainable: ~75K (adapter ~64K + FC expansion + aux_fc)
  vs DER's ~467K per task (~6x reduction)
"""

import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from models.base import BaseLearner
from utils.inc_net import DERLiteNet
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class DER_Lite(BaseLearner):
    """
    DER-Lite: Multi-scale adapters + KD, frozen backbone,
    single-stage training.
    Per-task trainable ~75K (vs DER's ~467K).
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = DERLiteNet(args, False)
        self._old_network = None

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0 and not isinstance(self._network, nn.DataParallel):
            self._network.freeze_backbone()
        self._old_network = self._network.copy().freeze()
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        logging.info("Total adapters: {}".format(len(self._network.adapters)))
        if len(self._network.adapters) > 0:
            logging.info(
                "Feature dim: {} (base={}, adapter_total={})".format(
                    self._network.feature_dim,
                    self._network.out_dim,
                    self._network.adapter_dim * len(self._network.adapters),
                )
            )

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(len(self._network.adapters) - 1):
                for p in self._network.adapters[i].parameters():
                    p.requires_grad = False
            self._network.freeze_backbone()

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network

        if self._cur_task >= 1 and len(self._network_module_ptr.adapters) > 0:
            self._network_module_ptr.backbone.eval()
            self._network_module_ptr.adapters[-1].train()
            self._network_module_ptr.fc.train()
            self._network_module_ptr.aux_fc.train()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        from tqdm import tqdm

        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        from tqdm import tqdm

        logging.info("=" * 50)
        logging.info("Training adapter + fc (+ KD loss from old model)")
        logging.info("=" * 50)

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_kd = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]

                loss_clf = F.cross_entropy(logits, targets.long())

                if self._old_network is not None:
                    old_logits = self._old_network(inputs)["logits"]
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        old_logits,
                        T,
                    )
                    loss = loss_clf + loss_kd
                else:
                    loss = loss_clf
                    loss_kd = torch.tensor(0.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_kd += loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


# ---------------------------------------------------------------------------
# DER-Lite V6: MMD-based adaptive backbone training
# ---------------------------------------------------------------------------


def _compute_mmd(X_old, X_new):
    """
    Compute unbiased MMD^2 between two feature distributions
    using RBF kernel with median-distance bandwidth heuristic.

    Reference: Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
    """
    max_samples = 500
    if X_old.shape[0] > max_samples:
        idx = torch.randperm(X_old.shape[0])[:max_samples]
        X_old = X_old[idx]
    if X_new.shape[0] > max_samples:
        idx = torch.randperm(X_new.shape[0])[:max_samples]
        X_new = X_new[idx]

    combined = torch.cat([X_old, X_new], dim=0)
    dists = torch.cdist(combined, combined)
    sigma = torch.median(dists[dists > 0]).clamp(min=0.1)
    gamma = 1.0 / (2.0 * sigma * sigma)

    K_old = torch.exp(-gamma * torch.cdist(X_old, X_old) ** 2)
    K_new = torch.exp(-gamma * torch.cdist(X_new, X_new) ** 2)
    K_cross = torch.exp(-gamma * torch.cdist(X_old, X_new) ** 2)

    m, n = X_old.shape[0], X_new.shape[0]
    mmd2 = (K_old.sum() - K_old.trace()) / (m * (m - 1)) \
         + (K_new.sum() - K_new.trace()) / (n * (n - 1)) \
         - 2.0 * K_cross.mean()
    return mmd2.item()


class DER_Lite_V6(DER_Lite):
    """
    DER-Lite V6: Adaptive backbone training via MMD distribution-shift
    detection.

    Before each incremental task, computes MMD between old exemplar
    backbone features and new task backbone features. If the feature
    distribution has shifted significantly:

      - Unfreeze backbone → train backbone + adapter + FC (~470K params)
      - If no significant shift → keep backbone frozen (~75K params)

    Re-freezes backbone after every task.
    """

    def __init__(self, args):
        super().__init__(args)
        self._mmd_threshold = args.get("mmd_threshold", 0.5)
        self._backbone_unfrozen = False

    def _get_backbone_features(self, data_loader):
        """Extract backbone features (no adapters) from a data loader."""
        ptr = self._network.module if isinstance(self._network, nn.DataParallel) \
              else self._network
        ptr.backbone.eval()
        features = []
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    _, inputs, _ = batch
                else:
                    inputs, _ = batch
                inputs = inputs.to(self._device)
                feats = ptr.backbone(inputs)["features"]
                features.append(feats)
                if len(torch.cat(features)) > 2000:
                    break
        return torch.cat(features, dim=0)

    def _detect_distribution_shift(self, data_manager):
        """
        Detect whether new task data has significantly different backbone
        features from old exemplar data. Returns True if backbone should
        be unfrozen.
        """
        memory = self._get_memory()
        if memory is None:
            return False

        mem_data, mem_targets = memory
        from torch.utils.data import TensorDataset, DataLoader as _DL
        import numpy as np

        if isinstance(mem_data, np.ndarray):
            if mem_data.ndim == 4:
                mem_data = torch.from_numpy(mem_data).permute(0, 3, 1, 2).float() / 255.0
            else:
                mem_data = torch.from_numpy(mem_data).float()
        elif isinstance(mem_data, torch.Tensor):
            mem_data = mem_data.float()
            if mem_data.max() > 1.0:
                mem_data = mem_data / 255.0
        if isinstance(mem_targets, np.ndarray):
            mem_targets = torch.from_numpy(mem_targets)

        mem_dataset = TensorDataset(mem_data, mem_targets)
        mem_loader = _DL(mem_dataset, batch_size=batch_size, shuffle=True)
        
        new_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test",
        )
        new_loader = _DL(
            new_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self._num_workers
        )

        X_old = self._get_backbone_features(mem_loader)
        X_new = self._get_backbone_features(new_loader)

        mmd2 = _compute_mmd(X_old, X_new)

        logging.info("-" * 50)
        logging.info(f"Task {self._cur_task + 1} Shift Detection")
        logging.info(f"  Old samples: {X_old.shape[0]}, New: {X_new.shape[0]}")
        logging.info(f"  MMD^2: {mmd2:.4f} (threshold={self._mmd_threshold})")
        shift = mmd2 > self._mmd_threshold
        logging.info(f"  Decision: {'UNFREEZE' if shift else 'keep frozen'}")
        logging.info("-" * 50)
        return shift

    def _unfreeze_backbone(self):
        """Temporarily unfreeze backbone for this task."""
        ptr = self._network.module if isinstance(self._network, nn.DataParallel) \
              else self._network
        for p in ptr.backbone.parameters():
            p.requires_grad = True
        self._backbone_unfrozen = True
        logging.info("Backbone UNFROZEN for this task")

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(len(self._network.adapters) - 1):
                for p in self._network.adapters[i].parameters():
                    p.requires_grad = False
            self._network.freeze_backbone()

            # === MMD gating ===
            if self._detect_distribution_shift(data_manager):
                self._unfreeze_backbone()
            # ===================

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Re-freeze backbone after training
        if self._backbone_unfrozen:
            self._network.freeze_backbone()
            self._backbone_unfrozen = False

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network

        if self._cur_task >= 1 and len(self._network_module_ptr.adapters) > 0:
            if not self._backbone_unfrozen:
                self._network_module_ptr.backbone.eval()
            self._network_module_ptr.adapters[-1].train()
            self._network_module_ptr.fc.train()
            self._network_module_ptr.aux_fc.train()
