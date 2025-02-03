using System;
using OGameSim.Production;

namespace OGameSim.Entities
{
    public class Astrophysics : IUpgradable
    {
        public Astrophysics()
        {
            SetUpgradeCost();
        }

        public uint Level { get; private set; }

        public Resources UpgradeCost { get; private set; }

        public void Upgrade()
        {
            Level++;
            SetUpgradeCost();
        }

        private void SetUpgradeCost()
        {
            var commonCost = (ulong)Math.Floor(4000 * Math.Pow(1.75, Level));
            UpgradeCost = new(
                commonCost,
                (ulong)Math.Floor(8000 * Math.Pow(1.75, Level)),
                commonCost
            );
        }
    }
}
