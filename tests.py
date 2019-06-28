#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Unit tests for functional segments of app
'''

import unittest

class TestSkuPreprocessorMethods(unittest.TestCase):
    
    def test_run_scale_standarize_remove_trend(self):
        pass
    
    def test_remove_trend(self):
        pass
    
    def test_remove_columns_no_columns(self):
        pass

    def test_remove_columns_chosen_columns(self):
        pass
    
    def test_inverse_transform(self):
        pass
    
    def test_no_transformation_ops(self):
        pass

    def test_save_dataframe(self):
        pass
    
    def test_sanitize_dataset_no_crop(self):
        pass
    
    def test_sanitize_dataset_crop(self):
        pass
    
    
class TestTrendRemover(unittest.TestCase):
    
    def test_fit_transform_all_columns(self):
        pass
    
    def test_fit_transform_chosen_columns(self):
        pass
    
    def test_inverse_transform(self):
        pass    