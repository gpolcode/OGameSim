using OGameSim.Entities;
using MemoizationPlayer;

var player = new Player();
var result = Planner.Search(player, 10);
Console.WriteLine(result);
