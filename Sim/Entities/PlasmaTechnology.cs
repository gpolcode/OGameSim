using OGameSim.Models;

namespace OGameSim.Entities
{
    public class PlasmaTechnology : IUpgradable
    {
        public PlasmaTechnology()
        {
            SetUpgradeCost();
        }

        public uint Level { get; private set; }
        public ResourcesModifier Modifier { get; private set; }
        public Resources UpgradeCost { get; private set; }

        public void Upgrade()
        {
            Level++;
            SetUpgradeCost();
            Modifier = new(
                1 + (Level * 1 / 100),
                1 + (Level * 0.66m / 100),
                1 + (Level * 0.33m / 100)
            );
        }

        private void SetUpgradeCost()
        {
            var exponent = 2 * (Level + 1);
            UpgradeCost = new(2000 ^ exponent, 4000 ^ exponent, 1000 ^ exponent);
        }
    }
}
