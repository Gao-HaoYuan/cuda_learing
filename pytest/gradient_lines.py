from typing import List, Optional, Dict
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# ---- 统一层名 ----
def _clean(name: str) -> str:
    return name.replace(".weight", "").replace("module.", "")

# ---- 固定层顺序（x 轴） ----
def build_layer_order(model, only_weight: bool = True) -> List[str]:
    order = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if only_weight and "weight" not in name:
            continue
        order.append(_clean(name))
    return order

# ---- 获取当前 step 的各层梯度 ----
@torch.no_grad()
def collect_grad_by_order(model, order: List[str]) -> Dict[str, float]:
    grad_map = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if "weight" not in name:
            continue
        grad_map[_clean(name)] = p.grad.detach().data.norm(2).item()

    aligned = {lname: grad_map.get(lname, 0.0) for lname in order}
    return aligned


# ---- 日志函数：在同一图上画多条 step 曲线 ----
class LayerGradPlotter:
    def __init__(self, writer: SummaryWriter, layer_order: List[str]):
        self.writer = writer
        self.layer_order = layer_order
        self.step_records = []  # 每个元素是 (step, grad_dict)

    def record(self, step: int, grad_dict: Dict[str, float]):
        self.step_records.append((step, grad_dict))

    def flush_epoch(self, epoch: int, topk: int = None):
        """在同一张图中画出多个 step 的曲线"""
        if not self.step_records:
            return

        names = self.layer_order

        # 可选：Top-K 层（按最后一个 step 的梯度排序）
        if topk is not None and topk > 0 and len(names) > topk:
            last_grad = self.step_records[-1][1]
            topk_names = sorted(names, key=lambda n: last_grad.get(n, 0.0), reverse=True)[:topk]
            names = topk_names

        # 画图
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.35), 4.0))
        for step, grad_dict in self.step_records:
            vals = [grad_dict.get(n, 0.0) for n in names]
            ax.plot(range(len(vals)), vals, marker='o', linewidth=1, label=f"step {step}")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=75, ha='right')
        ax.set_ylabel("Gradient L2-Norm")
        ax.set_title(f"Layer Gradients — Epoch {epoch}")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()

        # 写入 TensorBoard
        self.writer.add_figure(f"LayerGradientLines/epoch_{epoch}", fig, global_step=epoch)
        plt.close(fig)

        # 清空记录，为下一个 epoch 做准备
        self.step_records.clear()