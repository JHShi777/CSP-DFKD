import copy
import random
import time

import datafree
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv
from datafree.utils import ImagePool, DataIter


def custom_cross_entropy(preds, target):
    return torch.mean(torch.sum(-target * preds.log_softmax(dim=-1), dim=-1))


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


class LinearInputHook(object):
    """Capture the input of the last linear layer as a generic penultimate feature."""
    def __init__(self, module):
        self.proto = None
        self.hook = module.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, inputs):
        proto = inputs[0]
        if isinstance(proto, (list, tuple)):
            proto = proto[0]
        if proto.dim() > 2:
            proto = torch.flatten(proto, 1)
        self.proto = proto

    def close(self):
        self.hook.remove()


class CSP(BaseSynthesis):
    def __init__(self, teacher, student, generator, num_classes, img_size,
                 init_dataset=None, g_steps=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1,
                 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed=False,
                 warmup=10, bn_mmt=0, bnt=30, oht=1.5,
                 cr_loop=1, g_life=50, g_loops=1, gwp_loops=10, dataset="cifar10",
                 curriculum=False, curriculum_warmup_epochs=20,
                 proto=0.0, div=0.0, teach=0.0, conf=0.0,
                 proto_momentum=0.9, proto_gate=0.7, div_gate=0.7,
                 curriculum_easy_ratio=0.25, curriculum_mid_ratio=0.60):
        super(CSP, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size
        self.g_steps = g_steps
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = int(synthesis_batch_size / cr_loop)
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup
        self.g_life = g_life
        self.bnt = bnt
        self.oht = oht
        self.g_loops = g_loops
        self.gwp_loops = gwp_loops
        self.dataset = dataset
        self.label_list = torch.LongTensor([i for i in range(self.num_classes)])

        self.curriculum = curriculum
        self.curriculum_warmup_epochs = max(int(curriculum_warmup_epochs), 1)
        self.proto = proto
        self.div = div
        self.teach = teach
        self.conf = conf
        self.proto_momentum = proto_momentum
        self.proto_gate = proto_gate
        self.div_gate = div_gate
        self.curriculum_easy_ratio = curriculum_easy_ratio
        self.curriculum_mid_ratio = curriculum_mid_ratio

        teacher_core = unwrap_model(self.teacher)
        last_linear = None
        for m in teacher_core.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        self.feature_hook = LinearInputHook(last_linear) if last_linear is not None else None
        self.feature_dim = getattr(last_linear, 'in_features', None) if last_linear is not None else None
        if self.feature_dim is not None:
            self.class_feature_bank = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
            self.class_feature_count = torch.zeros(self.num_classes, device=self.device)
        else:
            self.class_feature_bank = None
            self.class_feature_count = None

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

        if dataset in ["imagenet", "tiny_imagenet"]:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                normalizer,
            ])
        else:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])

    def jitter_and_flip(self, inputs_jit, lim=1. / 8., do_flip=True):
        lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
        flip = random.random() > 0.5
        if flip and do_flip:
            inputs_jit = torch.flip(inputs_jit, dims=(3,))
        return inputs_jit

    def _generator_reinit(self):
        g = unwrap_model(self.generator)
        self.generator = g.reinit().to(self.device).train()

    def _generator_reinit_le(self):
        unwrap_model(self.generator).re_init_le()

    def _get_curriculum_scales(self):
        if not self.curriculum:
            return {
                'bn': 1.0, 'oh': 1.0, 'adv': 1.0,
                'conf': 1.0, 'proto': 1.0, 'div': 1.0, 'teach': 1.0,
                'stage': 'full'
            }
        prog = min(max((self.ep - self.ep_start) / float(self.curriculum_warmup_epochs), 0.0), 1.0)
        if prog < self.curriculum_easy_ratio:
            return {'bn': 1.2, 'oh': 1.2, 'adv': 0.0, 'conf': 1.0, 'proto': 0.0, 'div': 0.0, 'teach': 0.0, 'stage': 'easy'}
        if prog < self.curriculum_mid_ratio:
            return {'bn': 1.0, 'oh': 1.0, 'adv': 0.5, 'conf': 1.0, 'proto': 0.6, 'div': 0.25, 'teach': 0.0, 'stage': 'mid'}
        return {'bn': 0.8, 'oh': 1.0, 'adv': 1.0, 'conf': 0.8, 'proto': 1.0, 'div': 1.0, 'teach': 1.0, 'stage': 'hard'}

    @torch.no_grad()
    def _update_feature_bank(self, feats, targets, probs):
        if self.class_feature_bank is None or feats is None:
            return
        conf, pred = probs.max(dim=1)
        mask = (pred == targets) & (conf > self.proto_gate)
        if mask.sum() == 0:
            return
        sel_feats = feats[mask]
        sel_targets = targets[mask]
        for cls in sel_targets.unique():
            cls = int(cls.item())
            cls_mask = sel_targets == cls
            cls_proto = F.normalize(sel_feats[cls_mask].mean(dim=0), dim=0)
            if self.class_feature_count[cls] == 0:
                self.class_feature_bank[cls] = cls_proto
            else:
                self.class_feature_bank[cls] = F.normalize(
                    self.proto_momentum * self.class_feature_bank[cls] + (1.0 - self.proto_momentum) * cls_proto,
                    dim=0
                )
            self.class_feature_count[cls] += cls_mask.sum().float()

    def _feature_loss(self, feats, targets):
        if self.class_feature_bank is None or feats is None:
            return torch.zeros(1, device=self.device)
        proto_norm = F.normalize(feats, dim=1)
        bank = self.class_feature_bank[targets]
        valid = self.class_feature_count[targets] > 0
        if valid.sum() == 0:
            return torch.zeros(1, device=self.device)
        return F.mse_loss(proto_norm[valid], bank[valid])

    def _diversity_loss(self, feats, targets, probs):
        if feats is None or feats.shape[0] < 2:
            return torch.zeros(1, device=self.device)
        conf, pred = probs.max(dim=1)
        mask = (pred == targets) & (conf > self.div_gate)
        if mask.sum() < 2:
            return torch.zeros(1, device=self.device)
        feats = F.normalize(feats[mask], dim=1)
        targets = targets[mask]
        uniq = targets.unique()
        losses = []
        for cls in uniq:
            cls_feats = feats[targets == cls]
            if cls_feats.shape[0] < 2:
                continue
            sim = torch.mm(cls_feats, cls_feats.t())
            eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            losses.append(sim[~eye].mean())
        if len(losses) == 0:
            return torch.zeros(1, device=self.device)
        return torch.stack(losses).mean()

    def _teaching_loss(self, s_out, t_out, targets):
        probs = F.softmax(t_out.detach(), dim=1)
        conf, pred = probs.max(dim=1)
        gate = ((pred == targets) & (conf > self.proto_gate)).float()
        if gate.sum() == 0:
            return torch.zeros(1, device=self.device)
        per_sample = kldiv(s_out, t_out.detach(), reduction='none').sum(1)
        return -(per_sample * gate).sum() / gate.sum()

    def synthesize(self, targets=None):
        start = time.time()
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6

        if (self.ep - self.ep_start) % self.g_life == 0 or self.ep % self.g_life == 0:
            self._generator_reinit()

        g_loops = self.gwp_loops if self.ep < self.ep_start else self.g_loops
        self.ep += 1
        bi_list = []
        if g_loops == 0:
            return None, 0, 0, 0
        if self.dataset == "imagenet":
            idx = torch.randperm(self.label_list.shape[0])
            self.label_list = self.label_list[idx]

        scales = self._get_curriculum_scales()
        running_metrics = {'bn': 0.0, 'oh': 0.0, 'adv': 0.0, 'conf': 0.0, 'proto': 0.0, 'div': 0.0, 'teach': 0.0, 'total': 0.0}
        count_metrics = 0

        for gs in range(g_loops):
            best_inputs = None
            self._generator_reinit_le()

            if self.dataset == "imagenet":
                targets, ys = self.generate_ys_in(cr=0.0, i=gs)
            else:
                targets, ys = self.generate_ys(cr=0.0)
            ys = ys.to(self.device)
            targets = targets.to(self.device)

            optimizer = torch.optim.Adam([{'params': self.generator.parameters()}], lr=self.lr_g, betas=[0.5, 0.999])

            for it in range(self.g_steps):
                inputs = self.generator(targets=targets)
                inputs_aug = self.aug(self.jitter_and_flip(inputs) if self.dataset == "imagenet" else inputs)

                t_out = self.teacher(inputs_aug)
                probs = F.softmax(t_out, dim=1)
                teacher_feats = self.feature_hook.proto if self.feature_hook is not None else None

                loss_bn = sum([h.r_feature for h in self.hooks])
                loss_oh = custom_cross_entropy(t_out, ys.detach())
                loss_conf = F.nll_loss(torch.log(probs.clamp_min(1e-8)), targets)
                loss_proto = self._feature_loss(teacher_feats, targets)
                loss_div = self._diversity_loss(teacher_feats, targets, probs)

                if self.adv > 0 and (self.ep > self.ep_start):
                    s_out = self.student(inputs_aug)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
                    loss_teach = self._teaching_loss(s_out, t_out, targets)
                else:
                    s_out = None
                    loss_adv = loss_oh.new_zeros(1)
                    loss_teach = loss_oh.new_zeros(1)

                loss = (
                    self.bn * scales['bn'] * loss_bn +
                    self.oh * scales['oh'] * loss_oh +
                    self.adv * scales['adv'] * loss_adv +
                    self.conf * scales['conf'] * loss_conf +
                    self.proto * scales['proto'] * loss_proto +
                    self.div * scales['div'] * loss_div +
                    self.teach * scales['teach'] * loss_teach
                )

                with torch.no_grad():
                    self._update_feature_bank(teacher_feats, targets, probs)
                    if loss_oh.item() < best_oh:
                        best_oh = loss_oh
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.data
                    running_metrics['bn'] += float(loss_bn.detach().item())
                    running_metrics['oh'] += float(loss_oh.detach().item())
                    running_metrics['adv'] += float(loss_adv.detach().item())
                    running_metrics['conf'] += float(loss_conf.detach().item())
                    running_metrics['proto'] += float(loss_proto.detach().item())
                    running_metrics['div'] += float(loss_div.detach().item())
                    running_metrics['teach'] += float(loss_teach.detach().item())
                    running_metrics['total'] += float(loss.detach().item())
                    count_metrics += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.bn_mmt != 0:
                for h in self.hooks:
                    h.update_mmt()

            self.student.train()
            end = time.time()
            self.data_pool.add(best_inputs)
            bi_list.append(best_inputs)

            dst = self.data_pool.get_dataset(transform=self.transform)
            if self.init_dataset is not None:
                init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
                dst = torch.utils.data.ConcatDataset([dst, init_dst])
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
            loader = torch.utils.data.DataLoader(
                dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
                num_workers=4, pin_memory=True, sampler=train_sampler)
            self.data_iter = DataIter(loader)

        if count_metrics > 0:
            metrics = {k: v / count_metrics for k, v in running_metrics.items()}
        else:
            metrics = running_metrics
        metrics['stage'] = scales['stage']
        return {"synthetic": bi_list, "metrics": metrics}, end - start, best_cost, best_oh

    def sample(self):
        return self.data_iter.next()

    def generate_ys_in(self, cr=0.0, i=0):
        target = self.label_list[i * self.synthesis_batch_size:(i + 1) * self.synthesis_batch_size]
        target = torch.tensor([250, 230, 283, 282, 726, 895, 554, 555, 105, 107])
        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))
        return target, ys

    def generate_ys(self, cr=0.0):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for _ in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))
        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))
        return target, ys
