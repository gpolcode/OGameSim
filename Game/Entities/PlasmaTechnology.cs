using System;
using OGameSim.Production;

namespace OGameSim.Entities
{
    public class PlasmaTechnology : IUpgradable
    {
        public PlasmaTechnology()
        {
            SetUpgradeCost();
            SetModifier();
        }

        public uint Level { get; private set; }
        public ResourcesModifier Modifier { get; private set; }
        public Resources UpgradeCost { get; private set; }

        public void Upgrade()
        {
            Level++;
            SetUpgradeCost();
            SetModifier();
        }

        private void SetModifier()
        {
            Modifier = new(
                1 + (Level * 1m / 100),
                1 + (Level * 0.66m / 100),
                1 + (Level * 0.33m / 100)
            );
        }

        private void SetUpgradeCost()
        {
            UpgradeCost = new(
                2000 * (ulong)Math.Pow(2, Level),
                4000 * (ulong)Math.Pow(2, Level),
                1000 * (ulong)Math.Pow(2, Level)
            );
        }
    }
}
