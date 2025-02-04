using OGameSim.Production;

namespace OGameSim.Entities
{
    public interface IUpgradable
    {
        uint Level { get; }
        Resources UpgradeCost { get; }

        void Upgrade();
    }
}
