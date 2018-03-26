import pandas as pd
import numpy as np
import ray

class CoordDFBase(object):
    """Wrapper for Pandas indexes in Ray DataFrames. Handles all of the annoying
    metadata specific to the axis of partition (setting indexes, calculating the
    index within partition of a value, etc.) since the dataframe may be
    partitioned across either axis. This way we can unify the possible index
    operations over one axis-agnostic interface.

    IMPORTANT NOTE: Currently all operations are inplace.
    """

    def __init__(self, coord_df=None, lengths=None):
        self._coord_df = coord_df
        self._lengths = lengths

    ### Getters and Setters for Properties ###

    def _get__coord_df(self):
        if isinstance(self._coord_df_cache, ray.local_scheduler.ObjectID):
            self._coord_df_cache = ray.get(self._coord_df_cache)
        return self._coord_df_cache

    def _set__coord_df(self, coord_df):
        self._coord_df_cache = coord_df

    _coord_df = property(_get__coord_df, _set__coord_df)

    def _get_index(self):
        """Get the index wrapped by this IndexDF.

        Returns:
            The index wrapped by this IndexDF
        """
        return self._coord_df.index

    def _set_index(self, new_index):
        """Set the index wrapped by this IndexDF.

        Args:
            new_index: The new index to wrap
        """
        self._coord_df.index = new_index

    index = property(_get_index, _set_index)

    ### Accessor Methods ###

    def coords_of(self, key):
        raise NotImplementedError()

    def __getitem__(self, key):
        raise NotImplementedError()

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False, **kwargs):
        raise NotImplementedError()

    ### Modifier Methods ###

    def insert(self, key, loc=None, partition=None, index_within_partition=None):
        raise NotImplementedError()

    def drop(labels, errors='raise'):
        raise NotImplementedError()

    def rename_index(self, mapper):
        raise NotImplementedError()

    ### Factory Methods ###
    
    def from_index(index):
        """Defines an IndexDF from a Pandas Index
        
        Args:
            index (pd.Index): Index to wrap

        Returns: 
            An IndexDF backed by the specified pd.Index, with dummy partition data (?)
        """
        # NOTE: We decided to make an empty dataframe (pd.DataFrame(index=pd.RangeIndex(...)))
        dummy_coord_df = pd.DataFrame(index=index)
        dummy_lengths = [len(index)]
        return CoordDF(dummy_coord_df, dummy_lengths)

    from_index = staticmethod(from_index)

    def from_partitions(dfs, index=None, axis=0):
        """Defines an IndexDF from Ray DataFrame partitions
        
        Args:
            dfs ([ObjectID]): ObjectIDs of dataframe partitions
            index (pd.Index): Index of the Ray DataFrame.
            axis: Axis of partition (0=row partitions, 1=column partitions)

        Returns: 
            An IndexDF backed by the specified pd.Index, partitioned off specified partitions
        """
        # NOTE: We should call this instead of manually calling _compute_lengths_and_index in
        #       the dataframe constructor
        # TODO: Make it work for axis=1 as well
        lengths_oid, coord_df_oid = _compute_lengths_and_index(dfs, index)
        return CoordDF(coord_df_oid, lengths_oid)

    from_partitions = staticmethod(from_partitions)

class CoordDF(CoordDFBase):
    """Wrapper for Pandas indexes in Ray DataFrames. Handles all of the annoying
    metadata specific to the axis of partition (setting indexes, calculating the
    index within partition of a value, etc.) since the dataframe may be
    partitioned across either axis. This way we can unify the possible index
    operations over one axis-agnostic interface.

    IMPORTANT NOTE: Currently all operations are inplace.
    """

    def __init__(self, coord_df=None, lengths=None):
        self._coord_df = coord_df
        self._lengths = lengths

    ### Getters and Setters for Properties ###

    def _get__coord_df(self):
        if isinstance(self._coord_df_cache, ray.local_scheduler.ObjectID):
            self._coord_df_cache = ray.get(self._coord_df_cache)
        return self._coord_df_cache

    def _set__coord_df(self, coord_df):
        self._coord_df_cache = coord_df

    _coord_df = property(_get__coord_df, _set__coord_df)

    def _get_index(self):
        """Get the index wrapped by this IndexDF.

        Returns:
            The index wrapped by this IndexDF
        """
        return self._coord_df.index

    def _set_index(self, new_index):
        """Set the index wrapped by this IndexDF.

        Args:
            new_index: The new index to wrap
        """
        self._coord_df.index = new_index

    index = property(_get_index, _set_index)

    def _get__lengths(self):
        if isinstance(self._lengths_cache, ray.local_scheduler.ObjectID) or \
                (isinstance(self._lengths_cache, list) and \
                 isinstance(self._lengths_cache[0], ray.local_scheduler.ObjectID)):
            self._lengths_cache = ray.get(self._lengths_cache)
        return self._lengths_cache

    def _set__lengths(self, lengths):
        self._lengths_cache = lengths

    _lengths = property(_get__lengths, _set__lengths)

    ### Accessor Methods ###

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
        return self._coord_df.loc[key]

    def __getitem__(self, key):
        return self.coords_of(key)

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False, **kwargs):
        assignments_df = self._coord_df.groupby(by=by, axis=axis, level=level,
                                                as_index=as_index, sort=sort,
                                                group_keys=group_keys,
                                                squeeze=squeeze, **kwargs)\
            .apply(lambda x: x[:])
        return assignments_df

    ### Modifier Methods ###

    def insert(self, key, loc=None, partition=None, index_within_partition=None):
        # Perform insert on a specific partition
        # Determine which partition to place it in, and where in that partition
        if loc is not None:
            cum_lens = np.cumsum(self._lengths)
            partition = np.digitize(loc, cum_lens)
            if partition >= len(cum_lens):
                if loc > cum_lens[-1]:
                    raise IndexError("index {0} is out of bounds".format(loc))
                else:
                    index_within_partition = self._lengths[-1]
            else:
                index_within_partition = loc - np.asscalar(np.concatenate(([0], cum_lens))[partition])

        result = self._coord_df.copy()

        # Generate new index
        new_index = self.index.insert(loc, key)

        # Shift indices in partition where we inserted column
        idx_locs = (self._coord_df.partition == partition) & \
                   (self._coord_df.index_within_partition == index_within_partition) 
        # TODO: Determine why self._coord_df{,_cache} are read-only
        _coord_df_copy = self._coord_df.copy()
        _coord_df_copy.loc[idx_locs, 'index_within_partition'] += 1

        # TODO: Determine if there's a better way to do a row-index insert in pandas,
        #       because this is very annoying/unsure of efficiency
        # Create new coord entry to insert
        coord_to_insert = pd.DataFrame({'partition': partition,
                                        'index_within_partition': index_within_partition},
                                       index=[key])

        # Insert into cached RangeIndex, and order by new column index
        self._coord_df = _coord_df_copy.append(coord_to_insert).loc[new_index]

    def drop(labels, errors='raise'):
        self._coord_df.drop(values, errors=errors, inplace=True)

    def rename_index(self, mapper):
        self._coord_df.rename_axis(mapper, axis=0, inplace=True)

class IndexCoordDF(CoordDFBase):
    """Wrapper for Pandas indexes in Ray DataFrames. Handles all of the annoying
    metadata specific to the axis of partition (setting indexes, calculating the
    index within partition of a value, etc.) since the dataframe may be
    partitioned across either axis. This way we can unify the possible index
    operations over one axis-agnostic interface.

    IMPORTANT NOTE: Currently all operations are inplace.
    """

    def __init__(self, index):
        self._coord_df = pd.DataFrame(index=index)

    ### Accessor Methods ###

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
        locs = self.index.get_loc(key)
        # locs may be a single int, a slice, or a boolean mask.
        # Convert here to iterable of integers
        loc_idxs = pd.RangeIndex(len(self.index))[locs]
        # TODO: Investigate "modify view/copy" warning
        ret_obj = self._coord_df.loc[key]
        ret_obj['partition'] = 0
        ret_obj['index_within_partition'] = loc_idxs
        return ret_obj

    def __getitem__(self, key):
        return self.coords_of(key)

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False, **kwargs):
        assignments_df = self._coord_df.groupby(by=by, axis=axis, level=level,
                                                as_index=as_index, sort=sort,
                                                group_keys=group_keys,
                                                squeeze=squeeze, **kwargs)\
            .apply(lambda x: x[:])
        return assignments_df

    ### Modifier Methods ###

    def insert(self, key, loc=None, partition=None, index_within_partition=None):
        # Perform insert on a specific partition
        # Determine which partition to place it in, and where in that partition
        if loc is not None:
            cum_lens = np.cumsum(self._lengths)
            partition = np.digitize(loc, cum_lens)
            if partition >= len(cum_lens):
                if loc > cum_lens[-1]:
                    raise IndexError("index {0} is out of bounds".format(loc))
                else:
                    index_within_partition = self._lengths[-1]
            else:
                index_within_partition = loc - np.asscalar(np.concatenate(([0], cum_lens))[partition])

        result = self._coord_df.copy()

        # Generate new index
        new_index = self.index.insert(loc, key)

        # Shift indices in partition where we inserted column
        idx_locs = (self._coord_df.partition == partition) & \
                   (self._coord_df.index_within_partition == index_within_partition) 
        # TODO: Determine why self._coord_df{,_cache} are read-only
        _coord_df_copy = self._coord_df.copy()
        _coord_df_copy.loc[idx_locs, 'index_within_partition'] += 1

        # TODO: Determine if there's a better way to do a row-index insert in pandas,
        #       because this is very annoying/unsure of efficiency
        # Create new coord entry to insert
        coord_to_insert = pd.DataFrame({'partition': partition,
                                        'index_within_partition': index_within_partition},
                                       index=[key])

        # Insert into cached RangeIndex, and order by new column index
        self._coord_df = _coord_df_copy.append(coord_to_insert).loc[new_index]

    def drop(labels, errors='raise'):
        self._coord_df.drop(values, errors=errors, inplace=True)

    def rename_index(self, mapper):
        self._coord_df.rename_axis(mapper, axis=0, inplace=True)

    ### Factory Methods ###
    
    def from_index(index):
        """Defines an IndexDF from a Pandas Index
        
        Args:
            index (pd.Index): Index to wrap

        Returns: 
            An IndexDF backed by the specified pd.Index, with dummy partition data (?)
        """
        # NOTE: We decided to make an empty dataframe (pd.DataFrame(index=pd.RangeIndex(...)))
        dummy_coord_df = pd.DataFrame(index=index)
        dummy_lengths = [len(index)]
        return CoordDF(dummy_coord_df, dummy_lengths)

    from_index = staticmethod(from_index)

    def from_partitions(dfs, index=None, axis=0):
        """Defines an IndexDF from Ray DataFrame partitions
        
        Args:
            dfs ([ObjectID]): ObjectIDs of dataframe partitions
            index (pd.Index): Index of the Ray DataFrame.
            axis: Axis of partition (0=row partitions, 1=column partitions)

        Returns: 
            An IndexDF backed by the specified pd.Index, partitioned off specified partitions
        """
        # NOTE: We should call this instead of manually calling _compute_lengths_and_index in
        #       the dataframe constructor
        # TODO: Make it work for axis=1 as well
        lengths_oid, coord_df_oid = _compute_lengths_and_index(dfs, index)
        return CoordDF(coord_df_oid, lengths_oid)

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
