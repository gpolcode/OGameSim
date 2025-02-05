using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

class DQNAgent
{
    private readonly Module<Tensor, Tensor> model;
    private readonly Module<Tensor, Tensor> targetModel;
    private readonly torch.optim.Optimizer optimizer;
    private readonly Random random = new Random();

    private const float gamma = 0.99f; // Discount factor
    private const float learningRate = 0.001f;
    private const int batchSize = 32;
    private const int memorySize = 10000;
    private const float epsilonDecay = 0.995f;
    private const float minEpsilon = 0.01f;

    private float epsilon = 1.0f;
    private List<(Tensor, int, float, Tensor, bool)> memory =
        new List<(Tensor, int, float, Tensor, bool)>();

    public DQNAgent()
    {
        // Set device (DirectML for AMD GPU)
        var device = torch.cuda.is_available() ? torch.device("cuda") : torch.device("cpu");

        // Define network structure
        model = new DeepQNetwork().to(device);
        targetModel = new DeepQNetwork().to(device);
        targetModel.load_state_dict(model.state_dict());
        targetModel.eval();

        // Optimizer
        optimizer = torch.optim.Adam(model.parameters(), learningRate);
    }

    public int SelectAction(Tensor state)
    {
        if (random.NextDouble() < epsilon)
        {
            return random.Next(20); // Random action (exploration)
        }
        else
        {
            var qValues = model.forward(state);
            return (int)qValues.argmax().to(torch.ScalarType.Int64).item<long>(); // Explicit cast to int
        }
    }

    public void StoreExperience(Tensor state, int action, float reward, Tensor nextState, bool done)
    {
        if (memory.Count >= memorySize)
        {
            memory.RemoveAt(0); // Keep memory size fixed
        }
        memory.Add((state, action, reward, nextState, done));
    }

    public void Train()
    {
        if (memory.Count < batchSize)
            return; // Not enough samples

        var batch = memory.OrderBy(_ => random.Next()).Take(batchSize).ToList();

        var states = cat(batch.Select(x => x.Item1.unsqueeze(0)).ToArray(), 0);
        var actions = tensor(batch.Select(x => (long)x.Item2).ToArray(), dtype: ScalarType.Int64); // Fix: Convert to Int64
        var rewards = tensor(
            batch.Select(x => (float)x.Item3).ToArray(),
            dtype: ScalarType.Float32
        ); // Fix: Convert to Float32
        var nextStates = cat(batch.Select(x => x.Item4.unsqueeze(0)).ToArray(), 0);
        var dones = tensor(
            batch.Select(x => x.Item5 ? 1.0f : 0.0f).ToArray(),
            dtype: ScalarType.Float32
        ); // Fix: Convert to Float32

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

    public DeepQNetwork()
        : base("DeepQNetwork")
    {
        layer1 = Linear(50, 64);
        activation1 = ReLU();
        layer2 = Linear(64, 32);
        activation2 = Tanh();
        layer3 = Linear(32, 20);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        input = layer1.forward(input);
        input = activation1.forward(input);
        input = layer2.forward(input);
        input = activation2.forward(input);
        input = layer3.forward(input);
        return input;
    }
}

class Program
{
    static void Main()
    {
        var agent = new DQNAgent();
        var device = torch.cuda.is_available() ? torch.device("cuda") : torch.device("cpu");

        // Example training loop
        for (int episode = 0; episode < 1000; episode++)
        {
            var state = torch.randn(new long[] { 50 }).to(device); // Random initial state
            bool done = false;
            int step = 0;
            float totalReward = 0f;

            while (!done && step < 200)
            {
                int action = agent.SelectAction(state);
                var nextState = torch.randn([50]).to(device); // Simulated next state
                float reward = Random.Shared.Next(1, 10); // Simulated reward
                done = Random.Shared.NextDouble() < 0.1; // 10% chance of episode ending

                agent.StoreExperience(state, action, reward, nextState, done);
                agent.Train();

                state = nextState;
                totalReward += reward;
                step++;
            }

            agent.UpdateTargetNetwork();
            Console.WriteLine($"Episode {episode + 1}, Total Reward: {totalReward}");
        }
    }
}
