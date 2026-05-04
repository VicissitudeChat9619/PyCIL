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

class DER_SVD(DER):
    """
    DER with SVD-based Effective Rank Dynamic Expansion
    
    基于 SVD 有效秩的动态扩展机制：
    - 阶段1训练完新的特征提取器后，对特征矩阵进行 SVD 分解
    - 计算有效秩（Effective Rank），衡量特征的维度多样性
    - 若 有效秩 < threshold：判定为冗余，丢弃新的特征提取器
    - 若 有效秩 >= threshold：判定为有效新增，保留
    - 阶段2继续训练 fc（分类器）
    """
    def __init__(self, args):
        super().__init__(args)
        self.svd_threshold = args.get('svd_threshold', 10.0)  # 有效秩阈值
        logging.info(f"SVD threshold (effective rank): {self.svd_threshold}")

    def _compute_effective_rank(self, data_loader):
        """
        计算新特征提取器的有效秩（Effective Rank）
        
        有效秩计算方法：
        1. 对特征矩阵进行 SVD 分解
        2. 计算奇异值的归一化分布（使用 softmax）
        3. 使用熵公式：effective_rank = exp(-sum(p * log(p)))
        
        Returns:
            effective_rank: 有效秩值
        """
        self._network.eval()
        
        # 收集样本特征
        with torch.no_grad():
            all_inputs = []
            for _, (_, inputs, _) in enumerate(data_loader):
                all_inputs.append(inputs.to(self._device))
            all_inputs = torch.cat(all_inputs, dim=0)
        
        # 随机采样
        max_samples = 1000
        if len(all_inputs) > max_samples:
            indices = torch.randperm(len(all_inputs))[:max_samples]
            sampled_inputs = all_inputs[indices]
        else:
            sampled_inputs = all_inputs
        
        # 获取新骨干的特征
        features = self._network.convnets[-1](sampled_inputs)["features"]
        
        # SVD 分解
        # features: (N, D) -> SVD -> U(N,N), S(N), V(D,D)
        # 只需奇异值 S
        try:
            _, s, _ = torch.svd(features.float())
        except:
            # 如果 SVD 失败，使用特征值方法
            cov = torch.mm(features.T, features) / features.size(0)
            eigvals = torch.linalg.eigvalsh(cov)
            s = torch.sqrt(torch.clamp(eigvals.flip(0), min=0))
        
        # 计算有效秩
        # 方法：将奇异值归一化为概率分布，计算熵
        s_normalized = s / (s.sum() + 1e-10)  # 归一化
        entropy = -torch.sum(s_normalized * torch.log(s_normalized + 1e-10))
        effective_rank = torch.exp(entropy).item()
        
        # 额外信息：奇异值分布
        top_k = min(5, len(s))
        top_singular_values = s[:top_k].detach().cpu().numpy()
        
        logging.info("-" * 50)
        logging.info(f"SVD Analysis for Convnet[{len(self._network.convnets)-1}]")
        logging.info(f"  Feature matrix shape: {features.shape}")
        logging.info(f"  Top {top_k} singular values: {top_singular_values}")
        logging.info(f"  Total singular values: {len(s)}")
        logging.info(f"  Entropy: {entropy.item():.4f}")
        logging.info(f"  Effective Rank: {effective_rank:.4f}")
        logging.info(f"  SVD Threshold: {self.svd_threshold}")
        logging.info("-" * 50)
        
        return effective_rank

    def _update_representation(self, train_loader, test_loader, 
                                optimizer_stage1, scheduler_stage1,
                                optimizer_stage2, scheduler_stage2):
        # ========== 阶段1：训练新骨干 + fc + aux_fc ==========
        logging.info("=" * 50)
        logging.info("Stage 1: Training convnet + fc + aux_fc")
        logging.info("=" * 50)
        
        # 保存阶段1开始时的fc状态（用于SVD检测后恢复）
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

        # ========== SVD 有效秩检测 ==========
        if self._cur_task > 0:
            effective_rank = self._compute_effective_rank(train_loader)
            
            logging.info("=" * 50)
            logging.info(f"Task {self._cur_task} SVD Decision: effective_rank={effective_rank:.4f}, threshold={self.svd_threshold}")
            
            if effective_rank < self.svd_threshold:
                # 判定为冗余，丢弃新骨干
                logging.info(f"Effective rank ({effective_rank:.4f}) < threshold ({self.svd_threshold}): REDUNDANT!")
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
                logging.info(f"Effective rank ({effective_rank:.4f}) >= threshold ({self.svd_threshold}): VALID!")
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
            self._network.fc.train()
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
        计算新骨干与所有历史骨干的 CKA 相似度
        返回：与所有历史骨干的最大CKA值
        """
        if len(self._network.convnets) < 2:
            return 0.0
        
        self._network.eval()
        
        # 收集所有样本的特征
        with torch.no_grad():
            all_inputs = []
            for _, (_, inputs, _) in enumerate(data_loader):
                all_inputs.append(inputs.to(self._device))
            all_inputs = torch.cat(all_inputs, dim=0)
        
        # 随机采样
        max_samples = 1000
        if len(all_inputs) > max_samples:
            indices = torch.randperm(len(all_inputs))[:max_samples]
            sampled_inputs = all_inputs[indices]
        else:
            sampled_inputs = all_inputs
        
        # 获取当前新骨干的特征
        features_curr = self._network.convnets[-1](sampled_inputs)["features"]
        
        # 与所有历史骨干分别计算CKA
        cka_scores = []
        num_historical = len(self._network.convnets) - 1
        
        logging.info("-" * 50)
        logging.info(f"CKA Analysis: New Convnet vs {num_historical} Historical Convnets")
        logging.info("-" * 50)
        
        for i, convnet in enumerate(self._network.convnets[:-1]):
            features_prev = convnet(sampled_inputs)["features"]
            
            # 如果特征维度不同，对齐到相同维度
            if features_prev.shape[1] != features_curr.shape[1]:
                min_dim = min(features_prev.shape[1], features_curr.shape[1])
                features_prev = features_prev[:, :min_dim]
                features_curr = features_curr[:, :min_dim]
            
            cka = compute_cka(features_prev, features_curr)
            cka_scores.append(cka)
            logging.info(f"  CKA(convnet_new, convnet_{i}): {cka:.4f}")
        
        logging.info("-" * 50)
        logging.info(f"Max CKA: {max(cka_scores):.4f}, Mean CKA: {sum(cka_scores)/len(cka_scores):.4f}")
        logging.info("-" * 50)
        
        # 返回最大CKA值（最严格的判断）
        return max(cka_scores)

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
            logging.info(f"Task {self._cur_task} CKA Decision: max_cka={cka_score:.4f}, threshold={self.cka_threshold}")
            
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

class DER_KL(DER):
    """
    DER with KL Divergence-based Dynamic Expansion
    
    基于 KL 散度的动态扩展机制：
    - 阶段1训练完新的特征提取器后，使用旧类别回放数据进行评估
    - 旧路径：使用旧特征提取器 + 旧分类器权重得到 logits
    - 新路径：使用新特征提取器 + 新分类器得到 logits
    - 计算两个分布之间的 KL 散度
    - 若 KL < threshold：判定新特征提取器冗余，丢弃
    - 若 KL >= threshold：判定有效，保留
    """
    def __init__(self, args):
        super().__init__(args)
        self.kl_threshold = args.get('kl_threshold', 0.01)
        self.kl_temperature = args.get('kl_temperature', 2.0)
        logging.info(f"KL threshold: {self.kl_threshold}, Temperature: {self.kl_temperature}")

    def _compute_kl_divergence(self):
        """
        计算旧路径和新路径之间的 KL 散度
        
        使用旧类别回放数据，对比：
        - 旧路径：旧特征提取器 + 旧分类器权重
        - 新路径：新特征提取器 + 新分类器
        
        Returns:
            kl_div: KL 散度值
        """
        if len(self._network.convnets) < 2:
            return 0.0
        
        self._network.eval()
        
        # 获取旧类别回放数据（仅包含旧类别）
        memory_x, memory_y = self._get_memory()
        if len(memory_x) == 0:
            logging.warning("No replay data available, using test data for KL computation")
            return 0.0
        
        # 转换数据格式
        if isinstance(memory_x, np.ndarray):
            # numpy 数组格式是 (N, H, W, C)，需要转换为 (N, C, H, W)
            if memory_x.ndim == 4:
                memory_x = torch.from_numpy(memory_x).permute(0, 3, 1, 2).float()
            else:
                memory_x = torch.from_numpy(memory_x).float()
        if isinstance(memory_y, np.ndarray):
            memory_y = torch.from_numpy(memory_y)
        
        # 创建临时数据集
        dataset = torch.utils.data.TensorDataset(memory_x, memory_y)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        # 保存 fc 权重以构建旧分类器
        old_fc_weight = self._network.fc.weight.data.clone()
        old_fc_bias = self._network.fc.bias.data.clone()
        
        kl_divergences = []
        T = self.kl_temperature
        
        # 历史特征提取器数量（不包含最新的）
        num_historical = len(self._network.convnets) - 1
        # 旧路径特征维度 = 历史convnets数量 * 单个convnet输出维度
        old_feature_dim = self._network.out_dim * num_historical
        
        logging.info("-" * 50)
        logging.info(f"KL Divergence Analysis (T={T})")
        logging.info(f"  Historical convnets: {num_historical}")
        logging.info(f"  All convnets: {len(self._network.convnets)}")
        logging.info(f"  Old feature dim: {old_feature_dim}")
        logging.info(f"  Full feature dim: {self._network.feature_dim}")
        logging.info("-" * 50)
        
        # 旧类别数量
        num_old_classes = self._known_classes
        
        with torch.no_grad():
            for _, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self._device)
                
                # ========== 旧路径 ==========
                # 使用所有历史特征提取器 (convnets[:-1])
                features_old = []
                for convnet in self._network.convnets[:-1]:
                    feat = convnet(inputs)["features"]
                    features_old.append(feat)
                features_old = torch.cat(features_old, dim=1)  # (N, old_feature_dim)
                
                # ========== 新路径 ==========
                # 使用所有特征提取器（包含新的 convnets）
                features_new = []
                for convnet in self._network.convnets:
                    feat = convnet(inputs)["features"]
                    features_new.append(feat)
                features_new = torch.cat(features_new, dim=1)  # (N, full_feature_dim)
                
                # ========== 计算 logits（使用旧路径特征维度对应的权重部分）==========
                # 旧路径 logits：只取旧类别部分，只使用历史特征维度对应的权重
                old_weight_part = old_fc_weight[:num_old_classes, :old_feature_dim]
                old_bias_part = old_fc_bias[:num_old_classes]
                logits_old = torch.mm(features_old, old_weight_part.t()) + old_bias_part  # (N, num_old_classes)
                
                # 新路径 logits：使用全fc层，但只取旧类别部分
                # 注意：fc权重的前 old_feature_dim 列对应历史特征
                # 但fc对完整特征操作，我们只取旧类别部分
                new_weight_for_old_classes = self._network.fc.weight[:num_old_classes, :old_feature_dim]
                new_bias_for_old_classes = self._network.fc.bias[:num_old_classes]
                logits_new = torch.mm(features_old, new_weight_for_old_classes.t()) + new_bias_for_old_classes  # (N, num_old_classes)
                
                # 应用温度缩放的 softmax
                p_old = F.softmax(logits_old / T, dim=-1)
                p_new = F.softmax(logits_new / T, dim=-1)
                
                # 计算 KL(p_new || p_old)
                # KL_div = sum(p_new * log(p_new / p_old))
                kl_div = F.kl_div(p_new.log(), p_old, reduction='batchmean')
                kl_div = kl_div * (T * T)  # 反向缩放以获得真实 KL 值
                kl_divergences.append(kl_div.item())
        
        mean_kl = np.mean(kl_divergences)
        
        logging.info(f"  Number of samples: {len(memory_x)}")
        logging.info(f"  KL divergences per batch: {[f'{k:.6f}' for k in kl_divergences]}")
        logging.info(f"  Mean KL divergence: {mean_kl:.6f}")
        logging.info(f"  KL Threshold: {self.kl_threshold}")
        logging.info("-" * 50)
        
        return mean_kl

    def _update_representation(self, train_loader, test_loader, 
                                optimizer_stage1, scheduler_stage1,
                                optimizer_stage2, scheduler_stage2):
        # ========== 阶段1：训练新骨干 + fc + aux_fc ==========
        logging.info("=" * 50)
        logging.info("Stage 1: Training convnet + fc + aux_fc")
        logging.info("=" * 50)
        
        # 保存阶段1开始时的fc状态（用于KL检测后恢复）
        fc_state_before_s1 = None
        if self._cur_task > 0:
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

        # ========== KL 散度检测 ==========
        if self._cur_task > 0:
            kl_score = self._compute_kl_divergence()
            
            logging.info("=" * 50)
            logging.info(f"Task {self._cur_task} KL Decision: kl_div={kl_score:.6f}, threshold={self.kl_threshold}")
            
            if kl_score < self.kl_threshold:
                # 判定为冗余，丢弃新骨干
                logging.info(f"KL divergence ({kl_score:.6f}) < threshold ({self.kl_threshold}): REDUNDANT!")
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
                
                # 更新 aux_fc
                if len(self._network.task_sizes) > 0:
                    self._network.task_sizes.pop()
                    new_task_size = self._network.task_sizes[-1] if self._network.task_sizes else 0
                    self._network.aux_fc = self._network.generate_fc(
                        self._network.out_dim, new_task_size + 1
                    ).to(self._device)
                    
                logging.info(f"Convnets count after discard: {len(self._network.convnets)}")
                logging.info(f"Feature dimension: {self._network.feature_dim}")
            else:
                logging.info(f"KL divergence ({kl_score:.6f}) >= threshold ({self.kl_threshold}): VALID!")
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
            self._network.fc.train()
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

