# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import functools
import typing

import numpy as np
import pandas as pd


def downcast_dtypes(df: pd.DataFrame):
    """Downcast dtypes of a DataFrame from 64bit to 32bit."""
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df


def ensure_directory_exists(func: typing.Callable = None, directory: str = None):
    """Decorator that checks if a given directory exists. If not, the directory is created."""
    if func is None:
        return functools.partial(ensure_directory_exists, directory=directory)

    @functools.wraps(func)
    def f(*args, **kwargs):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        return func(*args, **kwargs)
    return f
