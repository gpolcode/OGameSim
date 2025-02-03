using OGameSim.Production;

namespace OGameSim.Entities
{
    internal interface IUpgradable
    {
        uint Level { get; }
        Resources UpgradeCost { get; }

        void Upgrade();
    }
}
