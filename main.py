"""
Monster Rancher Optimisation: Youngest Max Stats Monster
This program takes the starting stats of a monster and optimises to find the fewest weeks in which it can maximise all its stats. A training program (or training programs) will be provided.
"""

import pandas as pd
import numpy as np

import gurobipy as gp
from gurobipy import GRB

try:
    # Read data from csv files into dataframes

    # (Highest possible?) starting values for stats
    initial_stat_values = pd.read_csv('starting-data.csv')

    # Input stat gains for all 164 possible weeks in D rank
    D_rank_weeks_data = pd.read_csv('D-rank-data.csv')

    # Input stat gains for all 164 possible weeks in B rank
    B_rank_weeks_data = pd.read_csv('B-rank-data.csv')

    # Input stat gains for all 164 possible weeks in S rank
    S_rank_weeks_data = pd.read_csv('S-rank-data.csv')

    # Create convenient tuplelists and dictionary for adding vars and constraints

    # List of rank names
    rank_names = list(["d", "b", "s"])

    # List of stat names
    stat_names = list(initial_stat_values.columns)

    # Tuplelist to contain labels for possible weeks in each rank
    week_labels = gp.tuplelist()

    # Dictionary to contain all possible stat gains
    stat_gains = {}

    # Check all ranks have the same number of possible weeks
    weeks_in_rank = S_rank_weeks_data.shape[0]
    assert (D_rank_weeks_data.shape[0] ==
            B_rank_weeks_data.shape[0] == weeks_in_rank)

    # Populate tuplelist and dictionary
    for rank in rank_names:
        for week in range(0, weeks_in_rank):

            # Add week labels to tuplelist
            week_label = (rank, week)
            week_labels.append(week_label)

            for stat in stat_names:

                # Add stat gain data to dictionary
                stat_gain_label = (rank, week, stat)
                if rank == "s":
                    stat_gain = S_rank_weeks_data.at[week, stat]
                elif rank == "b":
                    stat_gain = B_rank_weeks_data.at[week, stat]
                elif rank == "d":
                    stat_gain = D_rank_weeks_data.at[week, stat]
                stat_gains[stat_gain_label] = stat_gain

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
    index = 1
    for stat in stat_names:
        current_stat_gains = {}

        for (rank, week) in week_labels:
            current_stat_gains[rank, week] = stat_gains[rank, week, stat]

        # Stats start at starting values and are maximised when they reach 999
        model.addConstr((week_counts.prod(current_stat_gains)
                         >= 999 - initial_stat_values.at[0, stat]), name="max_stats_constraint")
        print(
            f"{index}: {stat} constraint added. ({999 - initial_stat_values.at[0, stat]})")
        index += 1

    # Negative counts of weeks are not possible. Note that variables are assumed by Gurobi to be non-negative
    # TODO: Write constraints to ensure this?

    # Solve the model
    model.optimize()

    # Print training programme options
    if model.Status == GRB.OPTIMAL:

        # Check vs manually calculated lower bound (23 weeks)
        objective = int(model.getObjective().getValue())
        assert (objective <= 23)

        print(
            f"\nThe youngest possible max stats monster is {objective} weeks.")

        # TODO: Quantify 'overtraining'?

        count = model.getAttr('X', week_counts)

        # Outline D rank weeks
        d_total = count.sum("d", '*')
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
        b_total = count.sum("b", '*')
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
        s_total = count.sum("s", '*')
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
