import torch
from enum import IntEnum

# ===========================================================================
# 1. Define Indices (The "Schema")
#    Using IntEnum allows you to use names in code, but ints in tensors.
# ===========================================================================

class GlobalIdx(IntEnum):
    """Indices for the global player state tensor."""
    # Resources
    Metal = 0
    Crystal = 1
    Deuterium = 2
    Energy = 3
    
    # Research Levels
    Astrophysics = 4
    Plasma = 5
    EnergyTech = 6
    CombustionDrive = 7
    
    # Player Stats
    Points = 8
    
    # Size marker (keep this last)
    SIZE = 9

class PlanetIdx(IntEnum):
    """Indices for the per-planet state tensor."""
    # Mines
    MetalMine = 0
    CrystalMine = 1
    DeutSynthesizer = 2
    SolarPlant = 3
    FusionReactor = 4
    
    # Facilities
    Robotics = 5
    Shipyard = 6
    Nanite = 7
    
    # Attributes
    Temperature = 8
    FieldsUsed = 9
    
    # Size marker
    SIZE = 10

# ===========================================================================
# 2. The State Wrapper (The "Coding Experience")
#    This class wraps the tensors and exposes typed properties.
#    IDEs like VS Code / PyCharm will autocomplete these properties.
# ===========================================================================

class OGameBatch:
    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")):
        self.batch_size = batch_size
        self.device = device
        self.max_planets = 15  # OGame max limit

        # ALL data lives in these two tensors. 
        # This ensures maximum cache locality and GPU throughput.
        
        # Shape: (Batch, GlobalFeatures)
        self.global_data = torch.zeros((batch_size, int(GlobalIdx.SIZE)), device=device)
        
        # Shape: (Batch, MaxPlanets, PlanetFeatures)
        self.planet_data = torch.zeros((batch_size, self.max_planets, int(PlanetIdx.SIZE)), device=device)

    # --- Global Properties (IntelliSense works here) ---

    @property
    def metal(self) -> torch.Tensor:
        """Current Metal resources for all players."""
        return self.global_data[:, GlobalIdx.Metal]

    @metal.setter
    def metal(self, value: torch.Tensor):
        self.global_data[:, GlobalIdx.Metal] = value

    @property
    def crystal(self) -> torch.Tensor:
        return self.global_data[:, GlobalIdx.Crystal]

    @property
    def deuterium(self) -> torch.Tensor:
        return self.global_data[:, GlobalIdx.Deuterium]

    @property
    def astrophysics(self) -> torch.Tensor:
        return self.global_data[:, GlobalIdx.Astrophysics]

    # --- Planet Accessors ---

    def get_planet_level(self, planet_index: int, building_idx: PlanetIdx) -> torch.Tensor:
        """Get a specific building level for a specific planet index across all envs."""
        return self.planet_data[:, planet_index, building_idx]

    @property
    def all_metal_mines(self) -> torch.Tensor:
        """Returns (Batch, MaxPlanets) tensor of all metal mine levels."""
        return self.planet_data[..., PlanetIdx.MetalMine]

    # --- Helper Methods ---

    def clone(self) -> 'OGameBatch':
        """Creates a deep copy of the state."""
        new_state = OGameBatch(self.batch_size, self.device)
        new_state.global_data = self.global_data.clone()
        new_state.planet_data = self.planet_data.clone()
        return new_state