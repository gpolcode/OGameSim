namespace OGameSim.Models
{
	public readonly struct Resources
	{
		public ulong Crystal { get; }
		public ulong Deuterium { get; }
		public ulong Metal { get; }

		public Resources(ulong metal, ulong crystal, ulong deuterium)
		{
			Metal = metal;
			Crystal = crystal;
			Deuterium = deuterium;
		}

		public static Resources operator +(Resources a, Resources b)
			=> new(a.Metal + b.Metal, a.Crystal + b.Crystal, a.Deuterium + b.Deuterium);

		public static Resources operator -(Resources a, Resources b)
			=> new(a.Metal - b.Metal, a.Crystal - b.Crystal, a.Deuterium - b.Deuterium);

		public bool CanSubtract(Resources other)
		{
			return ConvertToMetalValue() >= other.ConvertToMetalValue();
		}

		public ulong ConvertToMetalValue()
		{
			var metalValue = (ulong)(Metal * ResourceWeight.MetalValue);
			var crystalValue = (ulong)(Crystal * ResourceWeight.CrystalValue);
			var deuteriumValue = (ulong)(Deuterium * ResourceWeight.DeuteriumValue);

			return metalValue + crystalValue + deuteriumValue;
		}
	}
}