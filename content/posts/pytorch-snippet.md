+++
title = 'Pytorch Snippet'
date = 2024-05-28T00:26:21+08:00
weight = 1
# draft = true
tags = ["pytorch"]
# author = ["Me", "You"] # multiple authors
showtoc = true
tocopen = true
+++

## 检测device

``` py
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

## Reproducibility

``` py
def reproducibility(seed: int = 8848):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
```
