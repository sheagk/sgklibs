#!/usr/bin/env python

DEFAULT_TABLE_FORMAT = 'ascii.fixed_width'

def my_loadtxt(fname, 
        as_table=False, 
        as_dict=False, 
        table_format=DEFAULT_TABLE_FORMAT, 
        **kwargs):
    """
    read a collimated text file and return the data as requested.

    leverages :meth:`astropy.table.Table.read` to read the 
    actual text file, then does some post-processing to return
    it in a useful format.

    Args:
        fname:
            name of the input file to read 

        as_table:
            whether to return the data directly, with no 
            post processing

        as_dict:
            whether to return the data as a dictionary with
            the keys being the column names

        table_format:
            format of the table to read
        
        kwargs:
            passed to :meth:`astropy.table.Table.read`

    
    Returns:
        the data in `fname`, either as an :class:`astropy.table.Table`,
        as a dictionary, or (if both are `as_table` and `as_dict` are false)
        as a list of arrays in the order the columns appear.
    """

    import os
    if not os.path.isfile(fname):
        raise IOError("unable to find {} for reading".format(fname))

    from astropy.table import Table
    import numpy as np
    t = Table.read(fname, format=table_format, **kwargs)
    if as_table:
        return t
    elif as_dict:
        return dict([(key, np.array(t[key])) for key in t.colnames])
    else:
        return [np.array(t[key]) for key in t.colnames]


def my_savetxt(outname, list_dict_or_arrays, colnames=None, order_dict=None,
               dtypes=None, table_format=DEFAULT_TABLE_FORMAT, overwrite=True,
               backup=True, replace_nan=-2, **kwargs):
    """
    flexibly save data to a file in ascii format using astropy

    preserves integer dtypes; otherwise tries float and defaults 
    to unicode otherwise

    Args:
        outname: name of output file

        list_dict_or_arrays:  many possibilities, assumes 
            that you want to save M columns:

            * a numpy structured array with M dtypes where each dtype 
              is a column.  this will get converted to, then treated like, 
              a dictionary of arrays
            * a list where each M items is a len N array
              (or list, in which case they'll be cast to arrays)
            * an N x M array
            * a dictionary of where each entry is an len N array
              (or list, in which case they'll be cast to arrays)

        colnames:
            single item (if list_dict_or_arrays is 1xN or Nx1), 
            list (must be len M) or dictionary giving column names.
            otherwise, will be constructed as 'col0', 'col1', 'col2', etc.
            if a dictionary is passed in, colnames will be constructed 
            from colnames if given, and dictionary keys otherwise.  if 
            a dictionary is passed in and colnames is NOT a dictionary,
            then order_dict must be handed in to assing column_names
        
        dtypes: either a single item or a list or dictionary giving data
            types.  useful if data is mixed type.  note that these should
            be numpy datatypes, as I'll be doing np.array(ar).astype(dtype).
            can also pass in a single item for everything.  

            if None, then I'll

                1. check if I'm an int (leave alone if so)
                2. check if I can become a float
                3.  leave as a string

        order_dict:
            ignored unless list_dict_or_arrays is a dictionary, in which
            case this gives the order in which the columns should be saved.
            if this isn't given, they're saved in alphabetical order -- that'll
            at least get things like vx, vy, vz and x, y, z together
    """
    import numpy as np

    if not len(kwargs.get('delimiter', '|').strip()):
        raise Warning(
            "!! -- warning -- white-space delimiters can cause problems!  recommend using a character such as |")

    assert type(list_dict_or_arrays) in [list, dict, np.ndarray, tuple]
    if type(list_dict_or_arrays) == dict and order_dict is not None:
        assert type(order_dict) == dict
    if colnames is not None:
        assert type(colnames) in [dict, list, np.ndarray, str]
    if dtypes is not None:
        assert type(dtypes) in [dict, list, str, np.dtype]

    def nan_replace(ar):
        if ar.dtype.kind in ['U', 'S']:
            return ar
        else:
            msk = np.isnan(ar)
            ar[msk] = replace_nan
            return ar

    def get_as_array(vals, dts, idx, cname):
        if type(dts) == list:
            ar = np.array(vals).astype(dts[idx])
        elif type(dts) in [str, np.dtype]:
            try:
                # allow for some allowances if I pass in just one dtype
                ar = np.array(vals).astype(dts)
            except (ValueError, TypeError):
                ar = np.array(vals).astype('U')
        elif type(dts) == dict:
            ar = np.array(vals).astype(dts[cname])
        else:
            ar = np.array(vals)
            if ar.dtype.kind != 'i':
                try:
                    ar = ar.astype('f')
                except ValueError:  # it's a string
                    ar = ar.astype('U')
        ar = nan_replace(ar)
        return ar

    # OK, lots to do here cause I'm trying to handle every possible case

    # first, if I hand in a structured array, convert it to a dictionary
    # i'll then handle dictionaries later
    if type(list_dict_or_arrays) == np.ndarray:
        if list_dict_or_arrays.dtype.names is not None:
            names = list_dict_or_arrays.dtype.names
            list_dict_or_arrays = dict(
                [(n, list_dict_or_arrays[n]) for n in names])

    # ok, now if got passed in a length 1 array, everything is (relatively) easy:
    input_as_array = np.transpose(np.array(list_dict_or_arrays, copy=True))
    onecol = False
    # single array; assume that you want to save as column, cause one line for a long array is dumb
    if len(input_as_array.shape) == 1:
        onecol = True
    if len(input_as_array.shape) > 1:
        if input_as_array.shape[1] == 1:  # single column
            onecol = True

    if onecol:
        if colnames is None:
            header = 'col0'
        elif type(colnames) in [list, np.ndarray]:
            header = colnames[0]
        else:
            header = colnames

        if dtypes is not None:
            if type(dtypes) == list:
                dtypes = dtypes[0]

            input_as_array = input_as_array.astype(dtypes)

        if backup:
            backup_file(outname)
        np.savetxt(outname, input_as_array, fmt='%'+dtype_dict.get(
            input_as_array.dtype.kind, input_as_array.dtype.kind), header=header)
        print("Wrote a single column to {}".format(outname))

    else:
        from astropy.table import Table
        t = Table()

        # ok, now handle the cases where I have multple entries to pass in.
        # first, let's do the most common, a list/sequence of arrays
        if type(list_dict_or_arrays) in [list, tuple]:
            if colnames is None:
                colnames = ['col'+str(ii)
                            for ii in range(len(list_dict_or_arrays))]
            for ii in range(len(list_dict_or_arrays)):
                ar = get_as_array(
                    list_dict_or_arrays[ii], dtypes, ii, colnames[ii])
                t[colnames[ii]] = ar

        # now handle a dictionary, in which case I have to worry about the order
        elif type(list_dict_or_arrays) == dict:
            keys = list(list_dict_or_arrays.keys())
            if colnames is None:
                colnames = keys
            else:
                assert (order_dict is not None) or type(colnames) == dict

            if order_dict is not None:
                order = [order_dict[k] for k in keys]
                sorti = np.argsort(order)
                keys = keys[sorti]
                if type(colnames) != dict:
                    colnames = colnames[sorti]

            for ii, key in enumerate(keys):
                if type(colnames) in [list, np.ndarray]:
                    c = colnames[ii]
                else:
                    c = colnames[key]

                ar = get_as_array(list_dict_or_arrays[key], dtypes, ii, key)
                t[c] = ar

        # now handle a single array, where I want to save a collimated version
        elif type(list_dict_or_arrays) == np.ndarray:
            ncol = list_dict_or_arrays.shape[1]
            if colnames is None:
                colnames = ['col'+str(ii) for ii in range(ncol)]
            assert type(colnames) in [list, np.ndarray]
            for ii in range(list_dict_or_arrays.shape[1]):
                ar = get_as_array(
                    list_dict_or_arrays[:, ii], dtypes, ii, colnames[ii])
                t[colnames[ii]] = ar
        else:
            raise ValueError("Don't know how I got here!")

        if backup:
            backup_file(outname)
        t.write(outname, format=table_format, overwrite=overwrite, **kwargs)
        print("Saved a table with {} rows and {} columns to {}".format(
            t.columns[0].size, len(t.columns), outname))
