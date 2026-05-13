"""
Partial-DER (pDER): efficiency improvement of DER.

Key ideas from the paper "On the Stability-Plasticity Dilemma of
Class-Incremental Learning" (CVPR 2023):
- Lower layers of ResNet are inherently stable (high CKA even in naive)
- Only apply DER expansion on the upper subset of layers (Layer 4)
- Shared trunk (conv1 through layer3) runs once, branches at layer4
- ~65% GMACs reduction vs original DER
"""

import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.der import DER
from utils.inc_net import PDERNet
from utils.toolkit import count_parameters

EPSILON = 1e-8


class PDER(DER):
    def __init__(self, args):
        super(DER, self).__init__(args)
        self._network = PDERNet(args, False)

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
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
            for p in self._network.trunk.parameters():
                p.requires_grad = False

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
        self.train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=self._num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=self._num_workers
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
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()
            self._network_module_ptr.trunk.eval()
