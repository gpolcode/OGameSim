using System;
using System.Threading.Tasks;
using OGameSim.Entities;
using OGameSim.Production;
using OneOf;
using RLMatrix;

public sealed class OGameEnvironment : IEnvironmentAsync<float[]>
{
    private const int _stateSize = 614;
    private const int _maxSteps = 8000;

    private readonly float[] _state = new float[_stateSize];

    private int _stepCounter = 0;
    private Player player = new();

    public OneOf<int, (int, int)> stateSize { get; set; } = _stateSize;
    public int[] actionSize { get; set; } = [63];

    public Task<float[]> GetCurrentState()
    {
        return Task.FromResult(_state);
    }

    public Task Reset()
    {
        player = new();
        _stepCounter = 0;
        Array.Fill(_state, 0);
        UpdateState();

        return Task.CompletedTask;
    }

    public Task<(float, bool)> Step(int[] actionsIds)
    {
        var actionId = actionsIds[0];
        var reward = ApplyAction(player, actionId);
        UpdateState();

        _stepCounter++;

        if (_stepCounter > _maxSteps)
        {
            return Task.FromResult((0f, true));
        }

        return Task.FromResult((reward, false));
    }

    public static float ApplyAction(Player player, int action)
    {
        float Penalty()
        {
            return -1;
        }

        float TryUpgrade(IUpgradable upgradable)
        {
            var currentPoints = player.Points;

            if (player.TrySpendResources(upgradable.UpgradeCost))
            {
                upgradable.Upgrade();
                var gainedPoints = (float)(player.Points - currentPoints);
                return (float)Math.Log10(gainedPoints + 1);
            }

            return Penalty();
        }

        float ProceedToNextDay()
        {
            player.ProceedToNextDay();
            return 1;
        }

        var planetIndex = (int)Math.Floor(action / 3d) - 1;
        if (planetIndex > player.Planets.Count - 1)
        {
            return Penalty();
        }

        var reward = (action, action % 3) switch
        {
            (0, _) => ProceedToNextDay(),
            (1, _) => TryUpgrade(player.Astrophysics),
            (2, _) => TryUpgrade(player.PlasmaTechnology),
            (_, 0) => TryUpgrade(player.Planets[planetIndex].MetalMine),
            (_, 1) => TryUpgrade(player.Planets[planetIndex].CrystalMine),
            (_, 2) => TryUpgrade(player.Planets[planetIndex].DeuteriumSynthesizer),
            _ => throw new NotImplementedException(),
        };

        return reward;
    }

    public void UpdateState()
    {
        var currentIndex = 0;

        void SetStateValue(float value)
        {
            _state[currentIndex] = value;
            currentIndex++;
        }

        void AddResources(Resources resources)
        {
            SetStateValue(resources.Metal);
            SetStateValue(resources.Crystal);
            SetStateValue(resources.Deuterium);
        }

        void AddResourcesModifier(ResourcesModifier resourcesModifier)
        {
            SetStateValue((float)resourcesModifier.Metal);
            SetStateValue((float)resourcesModifier.Crystal);
            SetStateValue((float)resourcesModifier.Deuterium);
        }

        // Player
        AddResources(player.Resources);

        // Plasma
        SetStateValue(player.PlasmaTechnology.Level);
        AddResourcesModifier(player.PlasmaTechnology.Modifier);
        AddResources(player.PlasmaTechnology.UpgradeCost);

        // Astro
        SetStateValue(player.Astrophysics.Level);
        AddResources(player.Astrophysics.UpgradeCost);

        // Planets
        foreach (var planet in player.Planets)
        {
            // Metal
            SetStateValue(planet.MetalMine.Level);
            AddResources(planet.MetalMine.UpgradeCost);
            AddResources(planet.MetalMine.TodaysProduction);
            AddResources(planet.MetalMine.UpgradeIncreasePerDay);

            // Crystal
            SetStateValue(planet.CrystalMine.Level);
            AddResources(planet.CrystalMine.UpgradeCost);
            AddResources(planet.CrystalMine.TodaysProduction);
            AddResources(planet.CrystalMine.UpgradeIncreasePerDay);

            // Deut
            SetStateValue(planet.DeuteriumSynthesizer.Level);
            AddResources(planet.DeuteriumSynthesizer.UpgradeCost);
            AddResources(planet.DeuteriumSynthesizer.TodaysProduction);
            AddResources(planet.DeuteriumSynthesizer.UpgradeIncreasePerDay);
        }
    }
}
