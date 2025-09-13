import numpy as np
import torch
import ctypes
import sys
import pathlib
import pytest

np = pytest.importorskip("numpy")
pythonnet = pytest.importorskip("pythonnet")
from pythonnet import load

# Ensure ogame_env package is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ogame_env import foo_torch

# Load .NET runtime and assembly
try:
    load("coreclr", runtime_config="./pyTorchPlayer/runtimeconfig.json")
    import clr
    clr.AddReference("C:/Code/gpolcode/OGameSim/Game/bin/Release/net8.0/publish/Game")
    from OGameSim.Entities import Player as CSPlayer
    from OGameSim.Production import Foo as CSFoo
    from System import IntPtr
except Exception:  # pragma: no cover - skip if runtime missing
    pytest.skip(".NET runtime or assembly not available", allow_module_level=True)


def cs_update_state(player):
    arr = np.zeros(125, dtype=np.float64)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    net_ptr = IntPtr(ctypes.addressof(ptr.contents))
    CSFoo.UpdateState(player, net_ptr)
    return arr

def test_parity_simple():
    cs_player = CSPlayer()
    torch_player = foo_torch.Player()

    cs_state = cs_update_state(cs_player)
    torch_state = foo_torch.update_state(torch_player).cpu().numpy()
    assert np.allclose(cs_state, torch_state)

    for action in [0, 1, 2, 3, 4]:
        cs_result = CSFoo.ApplyAction(cs_player, action)
        reward_t, done_t = foo_torch.apply_action(torch_player, action)
        cs_state = cs_update_state(cs_player)
        torch_state = foo_torch.update_state(torch_player).cpu().numpy()
        assert abs(cs_result.Item1 - reward_t.item()) < 1e-5
        assert cs_result.Item2 == done_t
        assert np.allclose(cs_state, torch_state)
