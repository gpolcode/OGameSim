using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using OpenTelemetry;
using OpenTelemetry.Exporter;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddHostedService<TrainingHostedService>();

builder
    .Services.AddOpenTelemetry()
    .ConfigureResource(resource => resource.AddService(serviceName: "OGameSim"))
    .UseOtlpExporter(OtlpExportProtocol.HttpProtobuf, new("http://host.docker.internal:4318"))
    .WithMetrics(metrics =>
    {
        metrics
            .AddMeter(RLMatrix.Meters.MeterName)
            .AddMeter(RLMatrixPlayer.Meters.MeterName)
            .AddView("*", new Base2ExponentialBucketHistogramConfiguration());
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
app.Run();
