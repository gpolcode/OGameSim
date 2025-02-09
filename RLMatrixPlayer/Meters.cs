using System.Diagnostics.Metrics;

namespace RLMatrixPlayer;

public static class Meters
{
    public const string MeterName = "ogamesim";

    private static readonly Histogram<double> _points;

    static Meters()
    {
        var meter = new Meter(MeterName);

        _points = meter.CreateHistogram<double>("points");
    }

    public static void UpdatePoints(decimal points)
    {
        _points.Record((double)points);
    }
}
