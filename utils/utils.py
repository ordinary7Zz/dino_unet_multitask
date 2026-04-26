import torch
import numpy as np

def log_print(message, log_file=None):
    print(message, end='')
    if log_file is not None:
        log_file.write(message)
        log_file.flush() 

def set_stage_trainability(model, stage, unfreeze_all_in_stage3=False, unfreeze_last_n=2):
    """
    stage:
      1 -> freeze all DINO
      2 -> unfreeze last N transformer blocks
      3 -> optionally unfreeze all DINO
    """

    # ---------- Stage 1: freeze all DINO ----------
    for name, param in model.named_parameters():
        if name.startswith("dino"):
            param.requires_grad = False

    # ---------- Stage 2: unfreeze last N blocks ----------
    if stage >= 2:
        block_params = []

        for name, param in model.named_parameters():
            if name.startswith("dino") and "blocks" in name:
                # 提取 block index，例如 dino.model.blocks.11.xxx
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "blocks" and i + 1 < len(parts):
                        if parts[i + 1].isdigit():
                            block_idx = int(parts[i + 1])
                            block_params.append((block_idx, name, param))
                        break

        if len(block_params) == 0:
            raise RuntimeError(
                "❌ No transformer blocks found in DINO backbone. "
                "Check timm model structure."
            )

        max_block = max(b[0] for b in block_params)

        for block_idx, name, param in block_params:
            if block_idx >= max_block - (unfreeze_last_n - 1):
                param.requires_grad = True

    # ---------- Stage 3: optionally unfreeze all ----------
    if stage == 3 and unfreeze_all_in_stage3:
        for name, param in model.named_parameters():
            if name.startswith("dino"):
                param.requires_grad = True

def build_optimizer(model, head_lr, backbone_lr, weight_decay):
    # 把参数分成：backbone(可训练部分) vs 其他(head/decoder)
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "dino" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

def gla_params(train_label_path, gla_tau, log_file=None):
    # 统计训练集类别频率用于GLA损失函数
    if train_label_path:
        import json
        with open(train_label_path, 'r') as f:
            train_labels = json.load(f)
        if isinstance(train_labels, dict):
            items = list(train_labels.values())
        else:
            items = train_labels
        
        # 统计良恶性样本数量
        benign_count = sum(1 for item in items if item.get('malignancy') == 0)
        malignant_count = sum(1 for item in items if item.get('malignancy') == 1)
        total_malignancy = benign_count + malignant_count
        
        # 添加数值稳定性保护，防止除零错误
        epsilon = 1e-8
        if total_malignancy > 0:
            p_benign = (benign_count + epsilon) / (total_malignancy + 2 * epsilon)
            p_malignant = (malignant_count + epsilon) / (total_malignancy + 2 * epsilon)
        else:
            p_benign = 0.5
            p_malignant = 0.5
        
        log_print(f"Training set - Benign/Malignant: Benign={benign_count} ({p_benign:.2%}), Malignant={malignant_count} ({p_malignant:.2%})\n", log_file)
        
        # 自适应tau策略：根据类别不平衡程度调整
        imbalance_ratio = max(p_benign, p_malignant) / min(p_benign, p_malignant)
        
        if gla_tau is not None:
            gla_tau = gla_tau
            log_print(f"Using manual tau={gla_tau:.2f}\n", log_file)
        else:
            # 自适应tau公式：
            if imbalance_ratio < 1.2:  # 几乎平衡 (60:40以内)
                gla_tau = 0.3
            elif imbalance_ratio < 1.5:  # 轻度不平衡 (60:40到75:25)
                gla_tau = 0.5
            elif imbalance_ratio < 2.5:  # 中度不平衡 (75:25到2.5:1)
                gla_tau = 0.7
            else:  # 严重不平衡
                gla_tau = 1.0
            log_print(f"Auto-adjusted tau={gla_tau:.2f} (imbalance_ratio={imbalance_ratio:.2f})\n", log_file)
        
        log_print(f"GLA parameters: p_benign={p_benign:.4f}, p_malignant={p_malignant:.4f}, tau={gla_tau:.2f}\n", log_file)
        log_print(f"Logit adjustment magnitude: {abs(gla_tau * (np.log(p_malignant) - np.log(p_benign))):.4f}\n", log_file)
        
        # 统计TI-RADS类别分布（1-5类）
        tirads_counts = [0] * 5
        for item in items:
            tirads = item.get('tirads', -1)
            if 1 <= tirads <= 5:
                tirads_counts[tirads - 1] += 1

        total_tirads = sum(tirads_counts)
        if total_tirads > 0:
            # 添加数值稳定性保护，防止除零错误
            p_tirads = torch.tensor([(c + epsilon) / (total_tirads + 5 * epsilon) for c in tirads_counts], dtype=torch.float32)
        else:
            p_tirads = torch.ones(5, dtype=torch.float32) / 5.0

        log_print(f"Training set - TI-RADS distribution (1-5): {tirads_counts}\n", log_file)
        log_print(f"TI-RADS probabilities: {p_tirads.tolist()}\n", log_file)
    else:
        p_benign = 0.5
        p_malignant = 0.5
        p_tirads = torch.ones(5, dtype=torch.float32) / 5.0
        gla_tau = gla_tau if gla_tau is not None else 0.5
    return p_benign, p_malignant, p_tirads, gla_tau