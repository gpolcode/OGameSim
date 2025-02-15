namespace OGameSim.Production
{
    public readonly record struct ResourcesModifier(
        decimal Metal,
        decimal Crystal,
        decimal Deuterium
    )
    {
        public static ResourcesModifier operator -(ResourcesModifier a, ResourcesModifier b) =>
            new(a.Metal - b.Metal, a.Crystal - b.Crystal, a.Deuterium - b.Deuterium);
    }
}
