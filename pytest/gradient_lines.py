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

        # 动态放大图像尺寸
        width = max(10, len(names) * 0.4)       # 宽度按层数自动拉伸
        height = 5 + 0.1 * len(self.step_records)  # 每多几条线略增高一点
        fig, ax = plt.subplots(figsize=(width, height))


        # --- 画折线 ---
        for step, grad_dict in self.step_records:
            vals = [grad_dict.get(n, 0.0) for n in names]
            ax.plot(range(len(vals)), vals, marker='o', linewidth=1, label=f"step {step}")

        # --- 坐标轴和网格 ---
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=70, ha='right', fontsize=9)
        ax.set_ylabel("Gradient L2-Norm", fontsize=11)
        ax.set_title(f"Layer Gradients — Epoch {epoch}", fontsize=12, pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # --- 图例布局优化 ---
        # 1) 放到外侧右边，不压主图
        # 放在右侧外部，自动两列显示
        ax.legend(
            fontsize=9,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=False,
            ncol=2,                  # ✅ 每行显示两个 legend 条目
            columnspacing=1.0,       # 列间距
            handlelength=2.0,        # 线段长度
            handletextpad=0.6,       # 线段和文字间距
        )

        # --- 调整边距，让 x 轴文字完全显示 ---
        plt.subplots_adjust(left=0.08, right=0.80, bottom=0.25, top=0.90)

        # 写入 TensorBoard
        self.writer.add_figure(f"LayerGradientLines/epoch_{epoch}", fig, global_step=epoch)
        plt.close(fig)

        # 清空记录，为下一个 epoch 做准备
        self.step_records.clear()