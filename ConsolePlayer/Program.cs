using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OGameSim.Entities;
using OpenTelemetry;
using OpenTelemetry.Exporter;
using OpenTelemetry.Resources;

var builder = WebApplication.CreateBuilder(args);

builder
    .Services.AddOpenTelemetry()
    .ConfigureResource(resource => resource.AddService(serviceName: "MyDotNetApp"))
    .UseOtlpExporter(OtlpExportProtocol.HttpProtobuf, new("http://host.docker.internal:4318"))
    .WithMetrics(metrics =>
    {
        metrics.AddMeter("player");
    })
    .WithLogging(
        default,
        options =>
        {
            options.IncludeFormattedMessage = true;
            options.IncludeScopes = true;
            options.ParseStateValues = true;
        }
    )
    .WithTracing();

var app = builder.Build();

app.MapGet(
    "/",
    () =>
    {
        var player = new Player();
        var tracer = new ActivitySource("player");
        var logger = app.Services.GetRequiredService<ILogger<Program>>();

        void LogStateChange(string resourceType, uint level, int day, decimal points)
        {
            logger.Log(
                LogLevel.Information,
                "{resourceType} leveled up to {level} on day {day} now got {points} points",
                resourceType,
                level,
                day,
                points
            );
        }

        var afkNess = Random.Shared.Next(0, 14);
        for (int i = 0; i < (365 * 8 * 2); i++)
        {
            player.ProceedToNextDay();
            if (Random.Shared.Next(0, afkNess) == 0)
            {
                continue;
            }

            var spentResources = true;
            while (spentResources)
            {
                spentResources = false;

                var upgradables = player
                    .Planets.ToList()
                    .SelectMany(x => new List<IUpgradable>()
                    {
                        x.MetalMine,
                        x.CrystalMine,
                        x.DeuteriumSynthesizer,
                    })
                    .Concat([player.Astrophysics, player.PlasmaTechnology])
                    .ToArray();

                var index = Random.Shared.Next(0, upgradables.Length);
                var upgradable = upgradables[index];

                if (player.TrySpendResources(upgradable.UpgradeCost))
                {
                    upgradable.Upgrade();
                    LogStateChange(upgradable.GetType().Name, upgradable.Level, i, player.Points);
                    spentResources = true;
                }
            }
        }

        return "lul";
    }
);

app.Run();

// var id = Activity.Current?.Id;
// Activity.Current?.SetTag("SetTag", "parent");
// var tracer = new ActivitySource("MyDotNetApp");
// using (var activity = tracer.StartActivity("CustomTrace"))
// {
//     if (id != null)
//     {
//         activity?.SetParentId(id);
//     }
//     activity?.AddEvent(new ActivityEvent("sample activity event."));
//     activity?.SetTag("SetTag", "example");
//     Thread.Sleep(200);
//     var logger = app.Services.GetRequiredService<ILogger<Program>>();
//     logger.LogCritical("This is a log message sent to Loki!");
//     activity?.SetStatus(ActivityStatusCode.Ok);
//     Thread.Sleep(200);
// }
// var creator = new InitialGameStateCreator
// {
//     SimulationDays = 10000,
//     PlanetCount = 14,
//     PlanetMaxTemperatur = -120,
//     PlanetPosition = 1,
// };

// var state = creator.Create();
// var gameService = new GameService(state);
// IUpgradeStrategy upgradeStrategy = new RoiUpgradeStrategy(state);
// using var writer = new GameStateWriter(creator, state);

// for (var i = 0; i < creator.SimulationDays; i++)
// {
//     upgradeStrategy.FindAndBuildUpgrades();
//     writer.WriteCurrentState();
//     gameService.MoveToNextDay();
// }
