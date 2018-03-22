import pandas as pd

class IndexDF(object):
    """Wrapper for Pandas indexes in Ray DataFrames. Handles all of the annoying
    metadata specific to the axis of partition (setting indexes, calculating the
    index within partition of a value, etc.) since the dataframe may be
    partitioned across either axis. This way we can unify the possible index
    operations over one axis-agnostic interface.
    """

    def __init__(self, coord_df=None, index=None):
        self._coord_df = None

        if partition_df is not None:
            self._coord_df = coord_df
        elif index is not None:
            # NOTE: We decided to make an empty dataframe (pd.DataFrame(index=pd.RangeIndex(...)))
            # TODO: Decide whether to create a dummy partition_df or an ad-hoc
            #       impl for non-partitioned axis indexes (i.e. col index when
            #       row partitioned)
            pass

    def _get_index(self):
        """Get the index wrapped by this IndexDF.

        Returns:
            The index wrapped by this IndexDF
        """
        return None

    def _set_index(self, new_index):
        """Set the index wrapped by this IndexDF.

        Args:
            new_index: The new index to wrap
        """
        pass

    index = property(_get_index, _set_index)

    def coords_of(self, key):
        """Returns the coordinates (partition, index_within_partition) of the
        provided key in the index

        Args:
            key: item to get coordinates of

        Returns:
            Pandas object with the keys specified. If key is a single object
            it will be a pd.Series with items `partition` and
            `index_within_partition`, and if key is a slice it will be a
            pd.DataFrame with said items as columns.
        """
        # TODO: Decide whether to create a dummy partition_df or an ad-hoc
        #       impl for non-partitioned axis indexes (i.e. col index when
        #       row partitioned)
        pass

    def __getitem__(self, key):
        return self.coords_of(key)
    
    ### Factory Methods ###
    
    def from_index(index):
        """Defines an IndexDF from a Pandas Index
        
        Args:
            index (pd.Index): Index to wrap

        Returns: 
            An IndexDF backed by the specified pd.Index, with dummy partition data (?)
        """
        return None

    from_index = staticmethod(from_index)

    def from_partitions(dfs, index=None):
        """Defines an IndexDF from Ray DataFrame partitions
        
        Args:
            dfs ([ObjectID]): ObjectIDs of dataframe partitions
            index (pd.Index): Index of the Ray DataFrame.

        Returns: 
            An IndexDF backed by the specified pd.Index, partitioned off specified partitions
        """
        # NOTE: We should call this instead of manually calling _compute_lengths_and_index in
        #       the dataframe constructor
        return None

    from_partitions = staticmethod(from_partitions)


"""
1. Think about passing one of these objs when returning a new ray.DF with same indexes (think applymap).
   Copy is going to be necessary, Devin has ideas about this.
2. This obj should also encompass lengths. (i.e. from_partitions should be a replacement for 
   _compute_lengths_and_index)
3. Decided to completely encapsulate underlying data. Expose a limited, but expandable, API for accessing
   metadata.
4. We decided to make an empty dataframe when we're indexing the non-partition axis (pd.DataFrame(index=pd.RangeIndex(...)))
"""
