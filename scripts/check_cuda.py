"""Verify the actual GPU can execute bf16 matmul and backpropagation."""

import json
from pathlib import Path
import torch

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    loss = (x @ x.T).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    report = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
              'bf16_supported': torch.cuda.is_bf16_supported(), 'compiled_arches': torch.cuda.get_arch_list(),
              'bf16_matmul_backward': 'passed', 'total_MiB': torch.cuda.get_device_properties(0).total_memory/2**20}
    Path('runs').mkdir(exist_ok=True)
    Path('runs/cuda_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
