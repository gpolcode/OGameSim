namespace PlanningPlayer;

using System;
using System.Collections.Generic;
using System.Linq;
using OGameSim.Entities;

/// <summary>
/// Monte Carlo tree search planner using ROI based rollout policy
/// and UCT for node selection.
/// </summary>
public sealed class MctsPlanner
{
    readonly int _iterations;
    readonly int _maxDepth;
    readonly Random _random = new(0);
    const double Exploration = 1.41421356237; // sqrt(2)

    public MctsPlanner(int iterations, int maxDepth)
    {
        _iterations = iterations;
        _maxDepth = maxDepth;
    }

    /// <summary>
    /// Runs MCTS and returns the best plan found.
    /// </summary>
    public List<ActionCandidate> Plan(Player root, int horizon)
    {
        var rootState = root.DeepClone();
        var rootNode = new Node(rootState, null, null, 0);

        for (int i = 0; i < _iterations; i++)
        {
            var node = Select(rootNode, horizon);
            if (node.Depth < _maxDepth && node.State.Day < horizon && node.UntriedActions.Count > 0)
            {
                node = Expand(node, horizon);
            }
            var reward = Simulate(node.State.DeepClone(), horizon);
            Backpropagate(node, reward);
        }

        // Extract best plan from root following highest average value.
        // Continue greedily until the horizon so the returned plan covers
        // the full planning window.
        var plan = new List<ActionCandidate>();
        var currentNode = BestChild(rootNode);
        var simState = root.DeepClone();

        while (currentNode != null && currentNode.Action.HasValue)
        {
            var action = currentNode.Action.Value;
            plan.Add(action);
            Planner.Apply(simState, action);
            currentNode = BestChild(currentNode);
        }

        while (simState.Day < horizon)
        {
            var next = Planner.EnumerateActions(simState)[0];
            plan.Add(next);
            Planner.Apply(simState, next);
        }

        return plan;
    }

    Node Select(Node node, int horizon)
    {
        while (node.UntriedActions.Count == 0 && node.Children.Count > 0 && node.Depth < _maxDepth && node.State.Day < horizon)
        {
            node = node.Children.OrderByDescending(Uct).First();
        }
        return node;
    }

    Node Expand(Node node, int horizon)
    {
        if (node.UntriedActions.Count == 0) return node;
        // pick next action (could randomize but deterministic order is fine)
        var index = _random.Next(node.UntriedActions.Count);
        var action = node.UntriedActions[index];
        node.UntriedActions.RemoveAt(index);
        var childState = node.State.DeepClone();
        Planner.Apply(childState, action);
        var child = new Node(childState, node, action, node.Depth + action.TimeCost);
        node.Children.Add(child);
        return child;
    }

    double Simulate(Player state, int horizon)
    {
        var depth = 0;
        while (depth < _maxDepth && state.Day < horizon)
        {
            var actions = Planner.EnumerateActions(state);
            var action = actions[0]; // ROI based (best ROI first)
            Planner.Apply(state, action);
            depth += action.TimeCost;
        }
        return (double)state.Points;
    }

    void Backpropagate(Node? node, double reward)
    {
        while (node != null)
        {
            node.Visits++;
            node.TotalValue += reward;
            node = node.Parent;
        }
    }

    Node? BestChild(Node node)
    {
        return node.Children.Count == 0
            ? null
            : node.Children.OrderByDescending(c => c.Visits == 0 ? double.NegativeInfinity : c.TotalValue / c.Visits).First();
    }

    double Uct(Node node)
    {
        if (node.Visits == 0)
            return double.PositiveInfinity;
        return (node.TotalValue / node.Visits) + Exploration * Math.Sqrt(Math.Log(node.Parent!.Visits) / node.Visits);
    }

    sealed class Node
    {
        public Player State { get; }
        public Node? Parent { get; }
        public ActionCandidate? Action { get; }
        public List<Node> Children { get; } = new();
        public List<ActionCandidate> UntriedActions { get; }
        public int Visits { get; set; }
        public double TotalValue { get; set; }
        public int Depth { get; }

        public Node(Player state, Node? parent, ActionCandidate? action, int depth)
        {
            State = state;
            Parent = parent;
            Action = action;
            Depth = depth;
            UntriedActions = Planner.EnumerateActions(state);
        }
    }
}

