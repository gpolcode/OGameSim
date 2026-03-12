using OGameSim.Production;

namespace OGameSim.Entities
{
    public abstract class Mine : IUpgradable
    {
        protected Mine(Resources baseProduction)
        {
            TodaysProduction = baseProduction;
            UpgradeCost = CalculateUpgradeCost();
            UpgradeIncreasePerDay = CalculateUpgradedProduction() - TodaysProduction;
        }

        public void Upgrade()
        {
            Level++;
            TodaysProduction += UpgradeIncreasePerDay;
            UpgradeCost = CalculateUpgradeCost();
            UpgradeIncreasePerDay = CalculateUpgradedProduction() - TodaysProduction;
        }

        public uint Level { get; private set; }
        public Resources TodaysProduction { get; private set; }
        public Resources UpgradeCost { get; private set; }
        public Resources UpgradeIncreasePerDay { get; private set; }

        protected abstract Resources CalculateUpgradeCost();
        protected abstract Resources CalculateUpgradedProduction();
    }
}
