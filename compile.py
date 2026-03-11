import torch

from state import OGameBatch

torch._logging.set_logs(graph_code=True)

mod1 = OGameBatch()
mod1.compile(mode="reduce-overhead")
print(mod1(torch.randn(3, 3)))