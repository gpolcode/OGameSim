using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OGameSim.Entities;
using OGameSim.Production;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

class DQNAgent
{
    private readonly Module<Tensor, Tensor> model;
    private readonly Module<Tensor, Tensor> targetModel;
    private readonly optim.Optimizer optimizer;

    private const float gamma = 0.99f; // Discount factor
    private const float learningRate = 0.001f;
    private const int batchSize = 32;
    private const int memorySize = 10000;
    private const float epsilonDecay = 0.995f;
    private const float minEpsilon = 0.01f;
    private float epsilon = 1.0f;
    private static ConcurrentQueue<(Tensor, int, float, Tensor, bool)> globalMemory = new();
    public const int ActionCount = 63;

    public DQNAgent()
    {
        // Set device (DirectML for AMD GPU)
        var device = cuda.is_available() ? torch.device("cuda") : torch.device("cpu");

        // Define network structure
        model = new DeepQNetwork().to(device);
        targetModel = new DeepQNetwork().to(device);
        targetModel.load_state_dict(model.state_dict());
        targetModel.eval();

        // Optimizer
        optimizer = optim.Adam(model.parameters(), learningRate);
    }

    public int SelectAction(Tensor state)
    {
        if (Random.Shared.NextDouble() < epsilon)
        {
            return Random.Shared.Next(ActionCount); // exploration
        }
        else
        {
            var qValues = model.forward(state);
            return (int)qValues.argmax().to(ScalarType.Int64).item<long>(); // Explicit cast to int
        }
    }

    public void StoreExperience(Tensor state, int action, float reward, Tensor nextState, bool done)
    {
        if (globalMemory.Count >= memorySize)
        {
            globalMemory.TryDequeue(out _); // Remove oldest experience
        }
        globalMemory.Enqueue((state, action, reward, nextState, done));
    }

    public void Train()
    {
        if (globalMemory.Count < batchSize)
            return; // Not enough samples

        var batch = globalMemory.OrderBy(_ => Guid.NewGuid()).Take(batchSize).ToList(); // Random sampling

        var states = cat(batch.Select(x => x.Item1.unsqueeze(0)).ToArray(), 0)
            .to(ScalarType.Float32);

        var actions = tensor(batch.Select(x => (long)x.Item2).ToArray(), dtype: ScalarType.Int64);
        var rewards = tensor(batch.Select(x => x.Item3).ToArray(), dtype: ScalarType.Float32);
        var nextStates = cat(batch.Select(x => x.Item4.unsqueeze(0)).ToArray(), 0)
            .to(ScalarType.Float32);

        var dones = tensor(
            batch.Select(x => x.Item5 ? 1.0f : 0.0f).ToArray(),
            dtype: ScalarType.Float32
        );

        var qValues = model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1);
        var nextQValues = targetModel.forward(nextStates).max(1).values;
        var targetQValues = rewards + gamma * nextQValues * (1 - dones);

        var loss = functional.mse_loss(qValues, targetQValues.detach());

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epsilon > minEpsilon)
        {
            epsilon *= epsilonDecay;
        }
    }

    public void UpdateTargetNetwork()
    {
        targetModel.load_state_dict(model.state_dict());
    }
}

class DeepQNetwork : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> layer1;
    private readonly Module<Tensor, Tensor> activation1;
    private readonly Module<Tensor, Tensor> layer2;
    private readonly Module<Tensor, Tensor> activation2;
    private readonly Module<Tensor, Tensor> layer3;
    private readonly Module<Tensor, Tensor> activation3;
    private readonly Module<Tensor, Tensor> layer4;
    private readonly Module<Tensor, Tensor> activation4;
    private readonly Module<Tensor, Tensor> layer5;
    private readonly Module<Tensor, Tensor> activation5;
    private readonly Module<Tensor, Tensor> outputLayer;

    public DeepQNetwork()
        : base("DeepQNetwork")
    {
        layer1 = Linear(614, 1024);
        activation1 = ReLU();
        layer2 = Linear(1024, 512);
        activation2 = ReLU();
        layer3 = Linear(512, 256);
        activation3 = ReLU();
        layer4 = Linear(256, 128);
        activation4 = ReLU();
        layer5 = Linear(128, 64);
        activation5 = ReLU();
        outputLayer = Linear(64, DQNAgent.ActionCount);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        input = layer1.forward(input);
        input = activation1.forward(input);
        input = layer2.forward(input);
        input = activation2.forward(input);
        input = layer3.forward(input);
        input = activation3.forward(input);
        input = layer4.forward(input);
        input = activation4.forward(input);
        input = layer5.forward(input);
        input = activation5.forward(input);
        input = outputLayer.forward(input);
        return input; // No activation on output (raw Q-values for RL)
    }
}

public class Program
{
    static void Main()
    {
        var totalEpisodes = 1000;

        Parallel.For(
            0,
            totalEpisodes,
            new ParallelOptions() { MaxDegreeOfParallelism = 4 },
            episode =>
            {
                var localAgent = new DQNAgent();
                var player = new Player();
                var state = GetState(player);
                var done = false;

                while (!done)
                {
                    var action = localAgent.SelectAction(state);
                    var reward = ApplyAction(player, action);
                    var nextState = GetState(player);

                    done = player.Day == 8000;
                    localAgent.StoreExperience(state, action, reward, nextState, done);
                    localAgent.Train();

                    if (player.Day % 50 == 0)
                    {
                        Console.WriteLine($"{episode}        {player.Points}");
                    }

                    state = nextState;
                }

                localAgent.UpdateTargetNetwork();
            }
        );
    }

    public static float ApplyAction(Player player, int action)
    {
        float Penalty()
        {
            return Math.Min(-10, (float)player.Points / 100 * -1);
        }

        float TryUpgrade(IUpgradable upgradable)
        {
            var currentPoints = player.Points;

            if (player.TrySpendResources(upgradable.UpgradeCost))
            {
                upgradable.Upgrade();
                return (float)(player.Points - currentPoints);
            }

            return Penalty();
        }

        float ProceedToNextDay()
        {
            player.ProceedToNextDay();
            return 10;
        }

        var planetIndex = (int)Math.Floor((action - 1) / 3d);
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

    public static Tensor GetState(Player player)
    {
        var states = new List<float>();

        void AddResources(Resources resources)
        {
            states.Add(resources.Metal);
            states.Add(resources.Crystal);
            states.Add(resources.Deuterium);
        }

        void AddResourcesModifier(ResourcesModifier resourcesModifier)
        {
            states.Add((float)resourcesModifier.Metal);
            states.Add((float)resourcesModifier.Crystal);
            states.Add((float)resourcesModifier.Deuterium);
        }

        // Player
        AddResources(player.Resources);

        // Plasma
        states.Add(player.PlasmaTechnology.Level);
        AddResourcesModifier(player.PlasmaTechnology.Modifier);
        AddResources(player.PlasmaTechnology.UpgradeCost);

        // Astro
        states.Add(player.Astrophysics.Level);
        AddResources(player.Astrophysics.UpgradeCost);

        // Planets
        for (int i = 0; i < 20; i++)
        {
            if (i > player.Planets.Count - 1)
            {
                states.AddRange(Enumerable.Repeat(0f, 30));
                continue;
            }

            var planet = player.Planets[i];

            // Metal
            states.Add(planet.MetalMine.Level);
            AddResources(planet.MetalMine.UpgradeCost);
            AddResources(planet.MetalMine.TodaysProduction);
            AddResources(planet.MetalMine.UpgradeIncreasePerDay);

            // Crystal
            states.Add(planet.CrystalMine.Level);
            AddResources(planet.CrystalMine.UpgradeCost);
            AddResources(planet.CrystalMine.TodaysProduction);
            AddResources(planet.CrystalMine.UpgradeIncreasePerDay);

            // Deut
            states.Add(planet.DeuteriumSynthesizer.Level);
            AddResources(planet.DeuteriumSynthesizer.UpgradeCost);
            AddResources(planet.DeuteriumSynthesizer.TodaysProduction);
            AddResources(planet.DeuteriumSynthesizer.UpgradeIncreasePerDay);
        }

        if (states.Count != 614)
        {
            throw new NotImplementedException();
        }
        return tensor(states);
    }
}
