using System.IO;
using OGameSim.Entities;
using MemoizationPlayer;

using var writer = new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true };
Console.SetOut(writer);

var player = new Player();
var result = Planner.Search(player, 8000);
Console.WriteLine(result);
Console.Out.Flush();
