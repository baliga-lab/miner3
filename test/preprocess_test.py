#!/usr/bin/env python3
import sys
import os
import pytest

import pandas as pd
from miner import miner


def test_remove_null_rows_min_0_remove_ok():
    df = pd.DataFrame([[0, 1, 2], [1, 2, 3], [0, 0, 0], [4, 5, 6]])
    df2 = miner.remove_null_rows(df)
    assert 3 == df2.shape[0], "wrong number of rows"

def test_remove_null_rows_min_0_unchanged():
    df = pd.DataFrame([[0, 1, 2], [1, 2, 3], [1, 0, 1], [4, 5, 6]])
    df2 = miner.remove_null_rows(df)
    assert 4 == df2.shape[0], "wrong number of rows"

def test_remove_null_rows_min_negative_unchanged():
    df = pd.DataFrame([[0, 1, -2], [1, 2, 3], [0, 0, 0], [4, 5, 6]])
    df2 = miner.remove_null_rows(df)
    assert 4 == df2.shape[0], "wrong number of rows"

EPS = 0.001

def test_correct_batch_effects_tpm():
    # large means to trigger the TPM function
    df = pd.DataFrame([[4, 1, 2], [1, 2, 3], [4, 5, 6]])
    df2 = miner.correct_batch_effects(df, False)
    assert (3, 3) == df2.shape
    assert abs(df2.values[0, 0] - 1.0910894511799618) < EPS
    assert abs(df2.values[1, 0] - (-1.0)) < EPS
    assert abs(df2.values[2, 0] - (-1.0)) < EPS

    assert abs(df2.values[0, 1] - (-0.8728715609439697)) < EPS
    assert abs(df2.values[1, 1] - 0.0) < EPS
    assert abs(df2.values[2, 1] - 0.0) < EPS

    assert abs(df2.values[0, 2] - (-0.2182178902359925)) < EPS
    assert abs(df2.values[1, 2] - 1.0) < EPS
    assert abs(df2.values[2, 2] - 1.0) < EPS


def test_correct_batch_effects_no_tpm():
    # small means standard deviation
    df = pd.DataFrame([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    df2 = miner.correct_batch_effects(df, False)
    assert (3, 3) == df2.shape
    for i in range(3):
        for j in range(3):
            assert abs(df2.values[i, j] - (-0.8164965809277261)) < EPS


def test_convert_ids_orig():
    """
    Test convert_ids_orig() function. Only the rows that have a matching
    gene will make it through this.
    """
    # simulate the preprocess data loader
    raw_expression = miner.read_file_to_df('testdata/exp_data-001.csv')
    raw_expression_zero_filtered = miner.remove_null_rows(raw_expression)
    zscored_expression = miner.correct_batch_effects(raw_expression_zero_filtered, do_preprocess_tpm=True)

    exp, conv_table = miner.convert_ids_orig(zscored_expression, 'testdata/identifier_mappings.txt')
    assert (7, 3) == exp.shape


def test_preprocess_main_simple():
    exp, conv_table = miner.preprocess('testdata/exp_data-001.csv', 'testdata/identifier_mappings.txt')
    print(exp)
    assert (7, 3) == exp.shape
    #for i in range(3):
    #    for j in range(3):
    #        assert abs(exp.values[i, j] - (-0.8164965809277261)) < EPS

