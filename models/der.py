# Please note that the current implementation of DER only contains the dynamic expansion process, since masking and pruning are not implemented by the source repo.
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, tensor2numpy


def compute_cka(features1, features2):
    """
    计算 Linear CKA (Centered Kernel Alignment) 相似度
    基于 Gram 矩阵的 Frobenius 范数
    
    参考: Kornblith et al., ICML 2019
    
    Args:
        features1: 特征张量 (N, D1)
        features2: 特征张量 (N, D2)
    
    Returns:
        CKA 相似度分数
    """
    # 中心化处理
    features1 = features1 - features1.mean(dim=0, keepdim=True)
    features2 = features2 - features2.mean(dim=0, keepdim=True)
    
    # 计算 Gram 矩阵
    gram1 = torch.mm(features1, features1.t())  # (N, N)
    gram2 = torch.mm(features2, features2.t())  # (N, N)
    
    # 中心化 Gram 矩阵 (HSIC 的无偏估计)
    n = gram1.size(0)
    trace = torch.trace
    centering = torch.eye(n, n, device=gram1.device) - torch.ones(n, n, device=gram1.device) / n
    
    hsic = trace(torch.mm(torch.mm(gram1, centering), gram2))
    
    # 归一化
    var1 = trace(torch.mm(torch.mm(gram1, centering), gram1))
    var2 = trace(torch.mm(torch.mm(gram2, centering), gram2))
    
    cka = hsic / (torch.sqrt(var1 * var2) + 1e-10)
    
    return cka.item()

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


class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

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
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=self._num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=self._num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
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
            # 阶段1优化器：新骨干 + fc
            optimizer_stage1 = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler_stage1 = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer_stage1, milestones=milestones, gamma=lrate_decay
            )
            # 阶段2优化器：只包含 fc
            optimizer_stage2 = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.fc.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler_stage2 = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer_stage2, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, 
                                        optimizer_stage1, scheduler_stage1,
                                        optimizer_stage2, scheduler_stage2)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
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

    def _update_representation(self, train_loader, test_loader, 
                                optimizer_stage1, scheduler_stage1,
                                optimizer_stage2, scheduler_stage2):
        # ========== 阶段1：训练新骨干 + fc + aux_fc ==========
        logging.info("=" * 50)
        logging.info("Stage 1: Training convnet + fc + aux_fc")
        logging.info("=" * 50)
        
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                
                loss = F.cross_entropy(logits, targets.long())
                
                optimizer_stage1.zero_grad()
                loss.backward()
                optimizer_stage1.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler_stage1.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Stage1 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Stage1 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

        # ========== 阶段2：重新初始化 fc，只训练 fc ==========
        logging.info("=" * 50)
        logging.info("Stage 2: Reinitializing fc, training fc only")
        logging.info("=" * 50)
        
        # 重新初始化 fc
        self._network.fc = self._network.generate_fc(
            self._network.feature_dim, self._total_classes
        ).to(self._device)
        
        # 重新创建阶段2优化器（只包含新的 fc）
        optimizer_stage2.param_groups = []
        optimizer_stage2.add_param_group(
            {'params': filter(lambda p: p.requires_grad, self._network.fc.parameters())}
        )
        # 重置 scheduler 的 milestones
        scheduler_stage2 = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer_stage2, milestones=milestones, gamma=lrate_decay
        )
        
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                
                loss = F.cross_entropy(logits, targets.long())
                
                optimizer_stage2.zero_grad()
                loss.backward()
                optimizer_stage2.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler_stage2.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Stage2 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Stage2 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)



class DER_A(DER):
    """
    DER with CKA-based Dynamic Expansion
    
    基于 CKA (Centered Kernel Alignment) 的动态扩展机制：
    - 阶段1训练完新的特征提取器后，计算其与上一阶段提取器的 CKA 相似度
    - 若 CKA > threshold：判定为冗余，丢弃新的特征提取器
    - 若 CKA <= threshold：判定为有效新增，保留
    - 阶段2继续训练 fc（分类器）
    """
    def __init__(self, args):
        super().__init__(args)
        self.cka_threshold = args.get('cka_threshold', 0.85)
        logging.info(f"CKA threshold: {self.cka_threshold}")

    def _compute_cka_similarity(self, data_loader):
        """
        计算新骨干与上一个骨干的 CKA 相似度
        """
        if len(self._network.convnets) < 2:
            return 0.0
        
        self._network.eval()
        features_prev = []
        features_curr = []
        
        with torch.no_grad():
            for _, (_, inputs, _) in enumerate(data_loader):
                inputs = inputs.to(self._device)
                
                # 提取前一个骨干的特征
                feat_prev = self._network.convnets[-2](inputs)["features"]
                features_prev.append(feat_prev)
                
                # 提取当前骨干的特征
                feat_curr = self._network.convnets[-1](inputs)["features"]
                features_curr.append(feat_curr)
        
        features_prev = torch.cat(features_prev, dim=0)
        features_curr = torch.cat(features_curr, dim=0)
        
        # 如果特征维度不同，对齐到相同维度
        if features_prev.shape[1] != features_curr.shape[1]:
            min_dim = min(features_prev.shape[1], features_curr.shape[1])
            features_prev = features_prev[:, :min_dim]
            features_curr = features_curr[:, :min_dim]
        
        # 随机采样计算 CKA
        max_samples = 1000
        if len(features_prev) > max_samples:
            indices = torch.randperm(len(features_prev))[:max_samples]
            features_prev = features_prev[indices]
            features_curr = features_curr[indices]
        
        return compute_cka(features_prev, features_curr)

    def _update_representation(self, train_loader, test_loader, 
                                optimizer_stage1, scheduler_stage1,
                                optimizer_stage2, scheduler_stage2):
        # ========== 阶段1：训练新骨干 + fc + aux_fc ==========
        logging.info("=" * 50)
        logging.info("Stage 1: Training convnet + fc + aux_fc")
        logging.info("=" * 50)
        
        # 保存阶段1开始时的fc状态（用于CKA检测后恢复）
        fc_state_before_s1 = {
            'weight': self._network.fc.weight.data.clone(),
            'bias': self._network.fc.bias.data.clone() if self._network.fc.bias is not None else None,
            'out_features': self._network.fc.out_features
        }
        
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                
                loss = F.cross_entropy(logits, targets.long())
                
                optimizer_stage1.zero_grad()
                loss.backward()
                optimizer_stage1.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler_stage1.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Stage1 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Stage1 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

        # ========== CKA 检测 ==========
        if self._cur_task > 0:
            cka_score = self._compute_cka_similarity(train_loader)
            
            logging.info("=" * 50)
            logging.info(f"CKA Similarity (Task {self._cur_task}): {cka_score:.4f}")
            logging.info(f"CKA Threshold: {self.cka_threshold}")
            
            if cka_score > self.cka_threshold:
                # 判定为冗余，丢弃新骨干
                logging.info(f"CKA score ({cka_score:.4f}) > threshold ({self.cka_threshold}): REDUNDANT!")
                logging.info("Discarding the new convnet...")
                
                # 删除新添加的骨干
                self._network.convnets.pop()
                
                # 恢复 fc 到阶段1开始时的状态
                new_fc = self._network.generate_fc(
                    self._network.feature_dim, fc_state_before_s1['out_features']
                ).to(self._device)
                new_fc.weight.data = fc_state_before_s1['weight']
                if fc_state_before_s1['bias'] is not None:
                    new_fc.bias.data = fc_state_before_s1['bias']
                self._network.fc = new_fc
                
                # 更新 task_sizes
                if len(self._network.task_sizes) > 0:
                    self._network.task_sizes.pop()
                    
                logging.info(f"Convnets count after discard: {len(self._network.convnets)}")
                logging.info(f"Feature dimension: {self._network.feature_dim}")
            else:
                logging.info(f"CKA score ({cka_score:.4f}) <= threshold ({self.cka_threshold}): VALID!")
                logging.info("Keeping the new convnet.")
            logging.info("=" * 50)

        # ========== 阶段2：重新初始化 fc，只训练 fc ==========
        logging.info("=" * 50)
        logging.info("Stage 2: Reinitializing fc, training fc only")
        logging.info("=" * 50)
        
        # 重新初始化 fc
        self._network.fc = self._network.generate_fc(
            self._network.feature_dim, self._total_classes
        ).to(self._device)
        
        # 重新创建阶段2优化器（只包含新的 fc）
        optimizer_stage2.param_groups = []
        optimizer_stage2.add_param_group(
            {'params': filter(lambda p: p.requires_grad, self._network.fc.parameters())}
        )
        # 重置 scheduler 的 milestones
        scheduler_stage2 = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer_stage2, milestones=milestones, gamma=lrate_decay
        )
        
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                
                loss = F.cross_entropy(logits, targets.long())
                
                optimizer_stage2.zero_grad()
                loss.backward()
                optimizer_stage2.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler_stage2.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Stage2 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Stage2 Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)