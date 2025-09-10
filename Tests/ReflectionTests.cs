using System.Reflection;
using MemoizationPlayer;
using Xunit;

public class ReflectionTests
{
    private sealed class Source
    {
        public int WithSetter { get; set; }
        public int WithoutSetter { get { return 7; } }
        public int PrivateSetter { get; private set; } = 5;
    }

    private sealed class Target
    {
        public int WithSetter { get; set; }
        public int WithoutSetter { get; } = 1;
        public int PrivateSetter { get; private set; }
    }

    [Fact]
    public void CopyProperty_CopiesWhenGetterAndSetterPresent()
    {
        var source = new Source { WithSetter = 42 };
        var target = new Target();
        InvokeCopy(source, target, nameof(Source.WithSetter));
        Assert.Equal(42, target.WithSetter);
    }

    [Fact]
    public void CopyProperty_SkipsWhenSetterMissing()
    {
        var source = new Source();
        var target = new Target();
        InvokeCopy(source, target, nameof(Source.WithoutSetter));
        Assert.Equal(1, target.WithoutSetter);
    }

    [Fact]
    public void CopyProperty_UsesPrivateSetter()
    {
        var source = new Source();
        var target = new Target();
        InvokeCopy(source, target, nameof(Source.PrivateSetter));
        var prop = typeof(Target).GetProperty(nameof(Target.PrivateSetter), BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;
        Assert.Equal(5, prop.GetValue(target));
    }

    static void InvokeCopy(object source, object target, string name)
    {
        var method = typeof(PlayerExtensions).GetMethod("CopyProperty", BindingFlags.NonPublic | BindingFlags.Static)!;
        method.Invoke(null, new object[] { source, target, name });
    }
}
