1) Dataset：返回张量 + 轻量 meta（不要放整段音频）
    ```python
    class DNSDataset(data.Dataset):
        dataset_name = "DNS"

        def __init__(self, noise_list, clean_list):
            super().__init__()
            self.noise_list = noise_list
            half = len(clean_list) // 2
            self.clean_list = clean_list[:half]

        def __len__(self):
            return len(self.clean_list)

        def __getitem__(self, item):
            scale = max(random.uniform(0, 1), 0.05)
            snr = random.choice([-5, 0, 5, 10, 15])

            clean_path = self.clean_list[item]
            clean = sf.read(clean_path, dtype="float32")[0]

            noise_path = random.choice(self.noise_list)
            noise = sf.read(noise_path, dtype="float32")[0]

            noisy, clean, clean_rms = mix_data(clean, noise, snr, scale)

            # 张量（会被堆叠）
            noisy = torch.as_tensor(noisy, dtype=torch.float32)
            clean = torch.as_tensor(clean, dtype=torch.float32)
            clean_rms = torch.as_tensor(clean_rms, dtype=torch.float32)

            # 轻量元数据（仅标量/字符串）
            meta = dict(
                idx=item,
                snr=snr,
                scale=scale,
                clean_path=clean_path,
                noise_path=noise_path,
                clean_len=len(clean),
                noisy_len=len(noisy),
            )
            return noisy, clean, clean_rms, meta
    ```

2) 自定义 collate_fn：张量堆叠，meta 变成 list
    ```python
    from torch.utils.data._utils.collate import default_collate

    def collate_keep_meta(batch):
        noisy, clean, clean_rms, meta = zip(*batch)   # 长度 = batch_size
        return (
            default_collate(noisy),       # [B, T] 或 pad 后 [B, T_max]
            default_collate(clean),       # [B, T]
            default_collate(clean_rms),   # [B]
            list(meta),                   # 保持为 list[dict]，不做堆叠
        )
    ```


如果音频长度不一致，请在 collate_fn 里做 padding（这里略）。否则会出现 “stack expects each tensor to be equal size”。

3) DataLoader：指定 collate_fn
    ```python
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_keep_meta,   # 👈 关键
    )
    ```

4) 训练循环：根据“外部的梯度范数”决定是否打印 meta
    ```python
    for it, (noisy, clean, clean_rms, metas) in enumerate(train_loader):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        clean_rms = clean_rms.to(device, non_blocking=True)

        out = model(noisy)
        loss = criterion(out, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 计算梯度范数（示例：全局范数）
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > 100:  # 外部条件成立 -> 打印/写文件
            print(f"[Grad={total_norm:.2f}] dumping metas for iter {it}:")
            for m in metas:   # metas 是 list[dict]
                print(f"  idx={m['idx']}, snr={m['snr']}, scale={m['scale']}, "
                    f"clean={m['clean_path']}, noise={m['noise_path']}")
            # 或者写到文件
            # with open("hit.txt","a",encoding="utf-8") as f: ...

        optimizer.step()
    ```

这样：

Dataset 里生成的参数（snr/scale/路径/...）成功带到外部；

是否打印完全由外部控制；

DataLoader 正常；meta 不参与计算，也不进 GPU（只是个 list）。