import torch
from enum import IntEnum

class GlobalIdx(IntEnum):
    """Indices for the global player state tensor."""
    # Research Levels
    Astrophysics = 4
    PlasmaTechnology = 5
    
    # Player Stats
    Points = 8
    Resources = 8
    Day = 8
    
    # Size marker (keep this last)
    SIZE = 9

class PlanetIdx(IntEnum):
    """Indices for the per-planet state tensor."""
    MetalLevel = 0
    MetalCost = 0
    MetalTodaysProduction = 0
    MetalIncreasePerDay = 0

    CrystallLevel = 0
    CrystallCost = 0
    CrystallTodaysProduction = 0
    CrystallIncreasePerDay = 0

    DeuteriumSynthesizerLevel = 0
    DeuteriumSynthesizerCost = 0
    DeuteriumSynthesizerTodaysProduction = 0
    DeuteriumSynthesizerIncreasePerDay = 0    

    MaxTemperature = 6

    # Size marker
    SIZE = 10

class OGameBatch(torch.nn.Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.max_planets = 20
        
        # Shape: (Batch, GlobalFeatures)
        self.global_data = torch.zeros((batch_size, int(GlobalIdx.SIZE)))
        
        # Shape: (Batch, MaxPlanets, PlanetFeatures)
        self.planet_data = torch.zeros((batch_size, self.max_planets, int(PlanetIdx.SIZE)))

    def forward(self, x):
        return torch.nn.functional.relu(self.global_data)
    
    @property
    def metal(self) -> torch.Tensor:
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