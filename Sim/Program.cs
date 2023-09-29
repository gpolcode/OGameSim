using OGameSim.Services;

var creator = new InitialGameStateCreator
{
    SimulationDays = 10000,
    PlanetCount = 14,
    PlanetMaxTemperatur = -120,
    PlanetPosition = 1,
};

var state = creator.Create();
var gameService = new GameService(state);
IUpgradeStrategy upgradeStrategy = new RoiUpgradeStrategy(state);
using var writer = new GameStateWriter(creator, state);

for (var i = 0; i < creator.SimulationDays; i++)
{
    upgradeStrategy.FindAndBuildUpgrades();
    writer.WriteCurrentState();
    gameService.MoveToNextDay();
}