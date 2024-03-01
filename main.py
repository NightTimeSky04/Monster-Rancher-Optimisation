"""
Monster Rancher Optimisation: Youngest Max Stats Monster (Monster Rancher 2)

This program takes the maximum starting stats of a monster alongside data on stat gains from fights in D, B and S ranks, and finds the fewest weeks in which all the monster's stats may be maximised. A training programme (or training programmes) is provided in the output. Use it wisely.

The problem is framed as an integer linear program, with the objective of minimising the total number of weeks spent in training. At least one week must be spent in each of D and B ranks. In order to complete training, all the monster's stats must be maxed out at 999. This program is solved by creating and optimising a model using Gurobipy.

This question was originally handed to me by Tsmuji (jack-cooper), along with a manually calculated upper bound on the number of weeks. The data used was kindly provided by SmilingFaces96.
"""

import pandas as pd

import gurobipy as gp
from gurobipy import GRB


class MonsterTraining:
    def __init__(self, initial_stats_file: str, rank_names: list):
        if initial_stats_file[len(initial_stats_file) - 4:] == ".csv":
            # Starting stats for the monster
            self.initial_stats = pd.read_csv(initial_stats_file)

            # List of stat names
            self.stat_names = list(self.initial_stats.columns)

            # List of ranks specified in input
            self.rank_names = rank_names

            self.training_stats_by_rank = self.get_training_stats_from_files()
            self.stat_gains = self.get_stat_gains_as_dict()

            # List of labels for available weeks
            # Arbitrary which stat gain dict this originates from
            self.week_labels, _ = gp.multidict(
                self.stat_gains[self.stat_names[0]])
        else:
            print(
                "\nERROR: File containing initial stats data must have the extension '.csv'.\n")

    def get_stat_gains_as_dict(self):
        # Reorganisation of stat gain data for convenience when adding variables and constraints to the model

        # Dictionary to contain all stat gain data dictionaries, indexed by stat name
        stat_gains = {}

        for stat in self.stat_names:
            stat_gains_by_week = {}

            # Inner dictionaries containing data indexed by rank name and week id
            for rank in self.rank_names:
                weeks_in_rank = self.training_stats_by_rank[rank].shape[0]

                for week in range(0, weeks_in_rank):
                    # Data pulled from appropriate dataframe of imported data
                    stat_gains_by_week[rank,
                                       week] = self.training_stats_by_rank[rank].at[week, stat]

            stat_gains[stat] = stat_gains_by_week

        return stat_gains

    def get_training_stats_from_files(self):
        # Import possible stat gain combinations from files

        # Dictionary to contain all imported stat gain data
        # Indexed by rank as csv files containing data are organised by rank
        training_stats_by_rank = {}

        for rank in self.rank_names:
            # Read data from csv files into dataframes
            # Pandas dataframes provide convenient methods for data handling
            filename = rank.upper() + "-rank-data.csv"
            training_data = pd.read_csv(filename)
            training_stats_by_rank[rank] = training_data

        return training_stats_by_rank

    def print_output(self):
        # Display optimal objective value
        print(
            f"\nThe youngest possible max stats monster is {self.objective} weeks.")

        # Display training programme
        self.print_training_programme()

    def print_training_programme(self):

        # Obtain counts of possible weeks in the solution
        self.count = self.model.getAttr('X', self.week_counts)

        for rank in self.rank_names:
            # Total number of weeks spent in each rank
            weeks_trained = int(self.count.sum(rank, '*').getValue())
            print(
                f"\n--- {rank.upper()} rank ---\nTotal weeks: {weeks_trained}\n")

            # Week ID alongside the number of repetitions of that week, discarding unused weeks
            for (rank, week) in self.week_labels.select(rank, '*'):
                if self.count[rank, week] > 0:
                    print(f"Week {week}: {int(self.count[rank, week])}")

                    # Outline of stat gains in each week
                    description = ""
                    for stat in self.stat_names:
                        description += f"   {stat}: {self.stat_gains[stat][rank, week]}"

                    print(description)

    def youngest_max_stats_monster(self, upper_bound=0):
        try:
            # Initialise model
            self.model = gp.Model("youngest_max_stats_monster")

            # Add variables to the model
            # Each variable represents a count of occurrances of a particular week in the monster's training
            # All possible weeks in input data are considered
            self.week_counts = self.model.addVars(
                self.week_labels, vtype=GRB.INTEGER, name="count")

            # Add objective function to the model
            # Objective is to minimise total number of weeks
            self.model.setObjective(self.week_counts.sum(), GRB.MINIMIZE)

            # Add rank constraints to the model
            # Monsters must spend at least 1 week at each specified on their way to S rank
            self.model.addConstrs((self.week_counts.sum(rank, '*') >=
                                   1 for rank in self.rank_names[:len(self.rank_names) - 1]), name="rank_constraints")

            # Add maximised stats constraints to the model
            # Each stat is maximised when total gain from training exceeds the difference between 999 and the stat starting value
            self.model.addConstrs((self.week_counts.prod(
                self.stat_gains[stat]) >= 999 - self.initial_stats.at[0, stat] for stat in self.stat_names), name="max_stats_constraints")

            # Negative counts of weeks are not possible. Variables are assumed by Gurobi to be non-negative.
            # TODO: Write constraints to ensure this?

            # Optimise the model
            self.model.optimize()

            # Perform checks on solution
            if self.model.Status == GRB.OPTIMAL:

                # Check objective vs an upper bound (if specified)
                if upper_bound > 0:
                    self.objective = int(self.model.getObjective().getValue())
                    assert (self.objective <= upper_bound)

            else:
                print("\nNo optimal solutions found.")

            # TODO: Quantify 'overtraining'?

            # Display details of the solution
            self.print_output()

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError:
            print('Encountered an attribute error')


youngest_max_stats_monster = MonsterTraining(
    "starting-data.csv", ["d", "b", "s"]).youngest_max_stats_monster(upper_bound=23)
