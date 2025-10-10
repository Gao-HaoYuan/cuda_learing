# show_torch_seeds.py
import torch

def show_torch_seeds():
    print("==== PyTorch RNG Seed Info ====")

    # 1️⃣ CPU 全局生成器
    cpu_seed = torch.initial_seed()
    print(f"[CPU] default generator seed: {cpu_seed}")

    # 2️⃣ CUDA 默认生成器（仅当前设备）
    if torch.cuda.is_available():
        cuda_seed = torch.cuda.initial_seed()
        print(f"[CUDA] default device seed: {cuda_seed}")

        # 3️⃣ 每个 GPU 的默认生成器
        for i, gen in enumerate(torch.cuda.default_generators):
            print(f"    device {i} seed: {gen.initial_seed()}")
    else:
        print("[CUDA] not available")

    # 4️⃣ 自定义 Generator 示例
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(2024)
    print(f"[Custom] CPU generator seed: {g_cpu.initial_seed()}")

    if torch.cuda.is_available():
        g_cuda = torch.Generator(device="cuda")
        g_cuda.manual_seed(999)
        print(f"[Custom] CUDA generator seed: {g_cuda.initial_seed()}")

    # 5️⃣ RNG 状态信息（仅查看，不建议打印完整内容）
    print("\n==== RNG State Summary ====")
    print(f"CPU RNG state length: {len(torch.get_rng_state())} bytes")
    if torch.cuda.is_available():
        print(f"CUDA RNG state length: {len(torch.cuda.get_rng_state())} bytes")

    print("================================\n")


if __name__ == "__main__":
    # 先设置一个固定的全局种子（方便复现）
    torch.manual_seed(123)

    # 打印所有 seed
    show_torch_seeds()
