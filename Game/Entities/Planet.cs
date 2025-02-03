namespace OGameSim.Entities
{
    public class Planet
    {
        public MetalMine MetalMine { get; } = new();
        public CrystalMine CrystalMine { get; } = new();
        public DeuteriumSynthesizer DeuteriumSynthesizer { get; }

        public int MaxTemperature { get; }

        public Planet(int maxTemperature)
        {
            MaxTemperature = maxTemperature;
            DeuteriumSynthesizer = new(MaxTemperature);
        }
    }
}
