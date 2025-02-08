using System;
using OGameSim.Production;

namespace OGameSim.Entities
{
    public class PlasmaTechnology : IUpgradable
    {
        public PlasmaTechnology()
        {
            SetUpgradeCost();
            Modifier = CalculateModifier(Level);
            UpgradedModifier = CalculateModifier(Level + 1);
        }

        public uint Level { get; private set; }
        public ResourcesModifier Modifier { get; private set; }
        public ResourcesModifier UpgradedModifier { get; private set; }
        public Resources UpgradeCost { get; private set; }

        public void Upgrade()
        {
            Level++;
            SetUpgradeCost();
            Modifier = CalculateModifier(Level);
            UpgradedModifier = CalculateModifier(Level + 1);
        }

        private ResourcesModifier CalculateModifier(uint level)
        {
            return new(
                1 + (level * 1m / 100),
                1 + (level * 0.66m / 100),
                1 + (level * 0.33m / 100)
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
