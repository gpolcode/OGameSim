using System;
using System.Collections.Concurrent;
using System.Reflection;

namespace PlanningPlayer;

static class AccessorCache
{
    static readonly ConcurrentDictionary<(Type type, string name), PropertyInfo?> _props = new();
    static readonly ConcurrentDictionary<(Type type, string name), FieldInfo?> _fields = new();

    public static PropertyInfo? GetProperty(Type type, string name)
    {
        return _props.GetOrAdd((type, name), key =>
            key.type.GetProperty(key.name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy));
    }

    public static FieldInfo? GetField(Type type, string name)
    {
        return _fields.GetOrAdd((type, name), key =>
            key.type.GetField(key.name, BindingFlags.NonPublic | BindingFlags.Instance));
    }
}
