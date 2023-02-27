using System;
using System.IO;
using System.Linq;
using System.Threading;
using CsvHelper;
using OGameSim.Models;

namespace OGameSim.Services
{
	public class GameStateWriter : IDisposable
	{
		private readonly GameState _state;
		private readonly CsvWriter _writer;

		public GameStateWriter(InitialGameStateCreator creator, GameState state)
		{
			_state = state;
			var streamWriter = new StreamWriter($"{DateTime.Now.Ticks}.csv");
			_writer = new(streamWriter, Thread.CurrentThread.CurrentCulture);

			_writer.WriteHeader<InitialGameStateCreator>();
			_writer.NextRecord();
			_writer.WriteRecord(creator);
			_writer.NextRecord();
			_writer.WriteHeader<DayModel>();
		}

		public void Dispose()
		{
			_writer.Dispose();
		}

		public void WriteCurrentState()
		{
			DayModel model = new()
			{
				AverageMetalMineLevel = _state.Planets.Average(x => x.MetalMine.Level),
				AverageCrystalMineLevel = _state.Planets.Average(x => x.CrystalMine.Level),
				AverageDeuteriumSynthesizerLevel = _state.Planets.Average(x => x.DeuteriumSynthesizer.Level),
				Metal = _state.Resources.Metal,
				Crystal = _state.Resources.Crystal,
				Deuterium = _state.Resources.Deuterium,
				Points = _state.Points
			};

			_writer.NextRecord();
			_writer.WriteRecord(model);
		}
	}
}