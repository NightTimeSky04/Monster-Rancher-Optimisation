"""
Monster Rancher Optimisation: Youngest Max Stats Monster (Monster Rancher 2)

This program takes the maximum starting stats of a monster alongside data on stat gains from fights in D, B and S ranks, and finds the fewest weeks in which all the monster's stats may be maximised. A training programme (or training programmes) is provided in the output. Use it wisely.

The problem is framed as an integer linear program, with the objective of minimising the total number of weeks spent in training. At least one week must be spent in each of D and B ranks. In order to complete training, all the monster's stats must be maxed out at 999. This program is solved by creating and optimising a model using Gurobipy.

This question was originally handed to me by Tsmuji (jack-cooper), along with a manually calculated upper bound on the number of weeks. The data used was kindly provided by SmilingFaces96.
"""

import pandas as pd
import numpy as np

import gurobipy as gp
from gurobipy import GRB

try:
    # Read data from csv files into dataframes
    # Pandas dataframes provide convenient methods for data handling

    # Highest possible starting values for stats
    initial_stat_values = pd.read_csv('starting-data.csv')

    # Stat gains data for all 164 possible weeks in D rank
    D_rank_weeks_data = pd.read_csv('D-rank-data.csv')

    # Stat gains data for all 164 possible weeks in B rank
    B_rank_weeks_data = pd.read_csv('B-rank-data.csv')

    # Stat gains data for all 164 possible weeks in S rank
    S_rank_weeks_data = pd.read_csv('S-rank-data.csv')

    # List of rank names
    rank_names = list(["d", "b", "s"])

    # List of stat names
    stat_names = list(initial_stat_values.columns)

    # Convenient tuplelist and dictionary for adding vars and constraints

    # Tuplelist to contain labels for possible weeks in each rank
    week_labels = gp.tuplelist()

    # Dictionary to contain all possible stat gains
    stat_gains = {}

    # Check all ranks have the same total number of possible weeks
    weeks_in_rank = S_rank_weeks_data.shape[0]
    assert (D_rank_weeks_data.shape[0] ==
            B_rank_weeks_data.shape[0] == weeks_in_rank)

    # Populate tuplelist and dictionary
    for rank in rank_names:
        for week in range(0, weeks_in_rank):

            week_data = []

            for stat in stat_names:

                # Data fetched from appropriate dataframe
                if rank == "s":
                    week_data.append(S_rank_weeks_data.at[week, stat])
                elif rank == "b":
                    week_data.append(B_rank_weeks_data.at[week, stat])
                elif rank == "d":
                    week_data.append(D_rank_weeks_data.at[week, stat])

                stat_gains[rank, week] = week_data

    # Multidict time
    week_labels, Lif_gains, Pow_gains, Int_gains, Ski_gains, Spd_gains, Def_gains = gp.multidict(
        stat_gains)

    # The model

    # Initialise model
    model = gp.Model("youngest_max_stats_monster")

    # Add variables to the model
    # Each variable represents a count of occurrances of a particular week in the monster's training
    # All possible weeks in D, B and S rank are considered
    # Each week has a unique set of stat gains
    week_counts = model.addVars(week_labels, vtype=GRB.INTEGER, name="count")

    # Add objective function to the model
    # Objective is to minimise total number of weeks
    model.setObjective(week_counts.sum(), GRB.MINIMIZE)

    # Add rank constraints to the model
    # Monsters must spend at least 1 week at D rank and 1 week at B rank on their way to S rank
    model.addConstr(week_counts.sum("d", '*') >= 1, name="D_rank_constraint")
    model.addConstr(week_counts.sum("b", '*') >= 1, name="B_rank_constraint")

    # Add maximised stats constraints to the model
    # Each stat is maximised when total gain from training exceeds the difference between 999 and the stat starting value
    index = 1
    for stat in stat_names:
        current_stat_gains = {}

        for (rank, week) in week_labels:
            current_stat_gains[rank, week] = stat_gains[rank, week, stat]

        model.addConstr((week_counts.prod(current_stat_gains)
                         >= 999 - initial_stat_values.at[0, stat]), name="max_stats_constraint")
        print(
            f"{index}: {stat} constraint added. ({999 - initial_stat_values.at[0, stat]})")
        index += 1

    # Negative counts of weeks are not possible. Variables are assumed by Gurobi to be non-negative.
    # TODO: Write constraints to ensure this?

    # Optimise the model
    model.optimize()

    # Output to terminal
    if model.Status == GRB.OPTIMAL:

        # Check objective vs manually calculated lower bound (23 weeks)
        objective = int(model.getObjective().getValue())
        assert (objective <= 23)

        # Display optimal objective value
        print(
            f"\nThe youngest possible max stats monster is {objective} weeks.")

        # TODO: Quantify 'overtraining'?

        # Breakdown of solution (training programme)

        # Get details of the solution
        count = model.getAttr('X', week_counts)

        # Outline D rank weeks
        d_total = int(count.sum("d", '*').getValue())
        print(
            f"\nTraining programme breakdown:\n\n--- D rank ---\nTotal weeks: {d_total}\n")
        for (rank, week) in week_labels.select("d", '*'):
            if count[rank, week] > 0:
                print(
                    f"Week {week}: {int(count[rank, week])}")
                description = ""
                for stat in stat_names:
                    description += f"   {stat}: {stat_gains[rank, week, stat]}"
                print(description)

        # Outline B rank weeks
        b_total = int(count.sum("b", '*').getValue())
        print(
            f"\n--- B rank ---\nTotal weeks: {b_total}\n")
        for (rank, week) in week_labels.select("b", '*'):
            if count[rank, week] > 0:
                print(
                    f"Week {week}: {int(count[rank, week])}")
                description = ""
                for stat in stat_names:
                    description += f"   {stat}: {stat_gains[rank, week, stat]}"
                print(description)

        # Outline S rank weeks
        s_total = int(count.sum("s", '*').getValue())
        print(
            f"\n--- S rank ---\nTotal weeks: {s_total}\n")
        for (rank, week) in week_labels.select("s", '*'):
            if count[rank, week] > 0:
                print(
                    f"Week {week}: {int(count[rank, week])}")
                description = ""
                for stat in stat_names:
                    description += f"   {stat}: {stat_gains[rank, week, stat]}"
                print(description)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
