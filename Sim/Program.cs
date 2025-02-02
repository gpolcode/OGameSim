using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Threading;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Exporter;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

var builder = WebApplication.CreateBuilder(args);

builder
    .Services.AddOpenTelemetry()
    .ConfigureResource(resource => resource.AddService(serviceName: "MyDotNetApp"))
    .UseOtlpExporter(OtlpExportProtocol.HttpProtobuf, new("http://host.docker.internal:4318"))
    .WithMetrics(metrics =>
    {
        metrics
            .AddAspNetCoreInstrumentation()
            .AddHttpClientInstrumentation()
            .AddMeter("MyCustomMetrics");
    })
    .WithLogging()
    .WithTracing(tracing =>
    {
        tracing
            .AddAspNetCoreInstrumentation()
            .AddHttpClientInstrumentation()
            .AddSource("MyDotNetApp");
    });

var app = builder.Build();

app.MapGet(
    "/",
    () =>
    {
        using var meter = new Meter("MyCustomMetrics");
        var requestCounter = meter.CreateCounter<int>("MyCustomMetrics_total");
        requestCounter.Add(1);

        var id = Activity.Current?.Id;
        Activity.Current?.SetTag("SetTag", "parent");

        var tracer = new ActivitySource("MyDotNetApp");
        using (var activity = tracer.StartActivity("CustomTrace"))
        {
            if (id != null)
            {
                activity?.SetParentId(id);
            }

            activity?.AddEvent(new ActivityEvent("sample activity event."));

            activity?.SetTag("SetTag", "example");
            Thread.Sleep(200);

            var logger = app.Services.GetRequiredService<ILogger<Program>>();
            logger.LogCritical("This is a log message sent to Loki!");

            activity?.SetStatus(ActivityStatusCode.Ok);
            Thread.Sleep(200);
        }

        return "lul";
    }
);

app.Run();

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
