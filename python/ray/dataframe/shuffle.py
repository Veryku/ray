from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import ray
from threading import Thread


@ray.remote(num_cpus=2)
class ShuffleActor(object):

    def __init__(self, partition_data):
        """Actor for facilitating distributed dataframe shuffle operations.
        Each partition in a Ray DataFrame will have be wrapped by a ShuffleActor, and
        during a shuffle, a collection of ShuffleActors will shuffle data onto each
        other together.

        Args:
            partition_data (ObjectID): The ObjectID of the partition this 
                ShuffleActor is wrapped around.
        """
        self.incoming = []
        self.partition_data = partition_data
        self.index_of_self = None

    def shuffle(self, index, partition_assignments, index_of_self, *list_of_partitions):
        """Performs a shuffle on the dataframe partitions represented by
        this ShuffleActor.

        Args:
            index (pd.Series):
                Indices of the Ray DataFrame that our partition_data contains
            partition_assignments ([pd.Series]): 
                List of series, each of which represents the indices to be moved to the 
                respective new partitions
            index_of_self (int):
                The partition number/index of our partition_data
            list_of_partitions ([ShuffleActor]): 
                The other ShuffleActors in this current shuffle
        """
        self.index_of_self = index_of_self
        self.assign_and_send_data(index, partition_assignments, list(list_of_partitions))

    def assign_and_send_data(self, index, partition_assignments, list_of_partitions):
        """Performs a shuffle on the dataframe partitions represented by
        this ShuffleActor.

        Args:
            index (pandas.DataFrame or ray.DataFrame???): ???
            partition_assignments ([pd.Series]): 
                List of series, each of which represents the indices to be moved to the 
                respective new partitions
            list_of_partitions ([ShuffleActor]): 
                The other ShuffleActors in this current shuffle
        """

        def calc_send(i, indices_to_send, data_to_send):
            indices_to_send[i] = [idx
                                  for idx in partition_assignments[i]
                                  if idx in index.index]
            data_to_send[i] = \
                self.partition_data.loc[indices_to_send[i]]

        num_partitions = len(partition_assignments)
        # Re-index partition_data
        self.partition_data.index = index.index

        indices_to_send = [None] * num_partitions
        data_to_send = [None] * num_partitions

        # For each existing partition, calculate which new partition to send each row to
        threads = [Thread(target=calc_send, args=(i, indices_to_send, data_to_send))
                   for i in range(num_partitions)]
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Append data to other new partitions' ShuffleActor's `add_to_incoming` lists
        for i in range(num_partitions):
            if i == self.index_of_self:
                continue
            try:
                list_of_partitions[i].add_to_incoming.remote((self.index_of_self, data_to_send[i]))

                self.partition_data = \
                    self.partition_data.drop(indices_to_send[i])
            except KeyError:
                pass

    def add_to_incoming(self, data):
        """Add data to the list of data to be coalesced into. Note that `self.incoming` is a
        list of Pandas DataFrames, which will eventually all be concatenated together

        Args:
            data (pd.DataFrame): A DataFrame containing the rows to be coalesced
        """

        self.incoming.append(data)

    def apply_func(self, func, *args):
        """Coalesce all the incoming data and apply a function on it

        Args:
            func: Function to apply onto coalesced data dataframe.
        """
        # Add data that was already on ourselves to incoming data list
        self.incoming.append((self.index_of_self, self.partition_data))
        # Sort by partition index
        self.incoming.sort(key=lambda x: x[0])
        self.incoming = [x[1] for x in self.incoming]
        data = pd.concat(self.incoming, axis=1)

        if len(args) == 0:
            return func(data)
        else:
            return func(data, *args)
