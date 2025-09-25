import torch
from torch.nn import attention

print(torch.__version__)
print(torch.backends.cuda.matmul.allow_tf32)
print(torch.backends.cuda.flash_sdp_enabled())   # <-- important
print(torch.backends.cuda.mem_efficient_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
