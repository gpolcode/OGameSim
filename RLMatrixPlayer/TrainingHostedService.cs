using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using RLMatrix;
using RLMatrix.Agents.Common;

public sealed class TrainingHostedService : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        var opts = new DQNAgentOptions()
        {
            Depth = 4,
            EPS_DECAY = 1000,
            LR = 0.00001f,
            TAU = 0.005f,
            GAMMA = 0.95f,
        };

        const string saveDirectory = @"C:/Users/Elsahr/Downloads/agent_storage";

        var envs = Enumerable.Repeat(0, 25).Select(x => new OGameEnvironment()).ToList();
        var myAgent = new LocalDiscreteRolloutAgent<float[]>(opts, envs);

        var newestSaveDirectory = new DirectoryInfo(saveDirectory)
            .GetDirectories()
            .OrderByDescending(d => d.LastWriteTimeUtc).FirstOrDefault();

        if (newestSaveDirectory != null)
        {
            await myAgent.Load(newestSaveDirectory.ToString());
        }

        while (true)
        {
            var identifier = DateTime.Now.ToString("yyyy'_'MM'_'dd'_'HH'_'mm'_'ss");

            for (int i = 0; i < 8000 * 2; i++)
            {
                await myAgent.Step();
            }

            var dir = Path.Combine(saveDirectory, identifier);
            Directory.CreateDirectory(dir);
            await myAgent.Save(dir);
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
