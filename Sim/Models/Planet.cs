using OGameSim.Entities;

namespace OGameSim.Models
{
	public class Planet
	{
		public CrystalMine CrystalMine { get; } = new();
		public DeuteriumSynthesizer DeuteriumSynthesizer { get; }

		public int MaxTemperature { get; }
		public MetalMine MetalMine { get; } = new();
		public ulong Position { get; }

		public Planet(int maxTemperature, ulong position)
		{
			MaxTemperature = maxTemperature;
			Position = position;

			DeuteriumSynthesizer = new(this);
		}
	}
}