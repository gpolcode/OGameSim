using CsvHelper.Configuration.Attributes;

namespace OGameSim.Models
{
	public class DayModel
	{
		[Index(2)] public double AverageCrystalMineLevel { get; init; }

		[Index(3)] public double AverageDeuteriumSynthesizerLevel { get; init; }

		[Index(1)] public double AverageMetalMineLevel { get; init; }

		[Index(5)] public ulong Crystal { get; init; }

		[Index(6)] public ulong Deuterium { get; init; }

		[Index(4)] public ulong Metal { get; init; }

		[Index(0)] public double Points { get; init; }
	}
}