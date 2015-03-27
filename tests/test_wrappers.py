import unittest
import os
import sys
sys.path.append('..')


from rdm.db import DBConnection, DBContext, OrangeConverter, RSDConverter
from rdm.wrappers import Wordification, RSD
from conf import TEST_DB, RESULTS_FOLDER


class TestWrappers(unittest.TestCase):

    def setUp(self):
        self.connection = DBConnection(
            TEST_DB['user'],
            TEST_DB['pass'],
            TEST_DB['host'],
            TEST_DB['database']
        )
        self.context = DBContext(self.connection)
        self.context.target_table = 'trains'
        self.context.target_att = 'direction'


    def test_wordification(self):
        conv = OrangeConverter(self.context)
        wordification = Wordification(
            conv.target_Orange_table(),
            conv.other_Orange_tables(),
            self.context
        )
        wordification.run(1)
        wordification.calculate_weights()

        with open(os.path.join(RESULTS_FOLDER, 'wrappers', 'wordification', 'trains.arff')) as f:
            self.assertMultiLineEqual(wordification.to_arff(), f.read())


    def test_rsd(self):
        conv = RSDConverter(self.context)
        rsd = RSD()
        features, arff, _ = rsd.induce(conv.background_knowledge(), examples=conv.all_examples())

        with open(os.path.join(RESULTS_FOLDER, 'wrappers', 'rsd', 'trains.arff')) as f:
            self.assertMultiLineEqual(arff, f.read())

        with open(os.path.join(RESULTS_FOLDER, 'wrappers', 'rsd', 'trains.frs')) as f:
            self.assertMultiLineEqual(features, f.read())
