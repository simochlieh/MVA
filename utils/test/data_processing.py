"""
Unit tests for development purposes
"""
import mock
import utils.data_processing as dp
import unittest


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """
        Mocking data file
        """
        with mock.patch('__main__.open', mock.mock_open(), create=True) as m:
            with open("foo", 'w') as f:
                f.write("""-3.60340505	1.3266	1.0000
                -4.21901140	2.0150	1.0000
                -1.51565812	0.5059	1.0000
                -1.16975695	0.3815	1.0000
                0.52274116	-0.6572	1.0000
                -0.14174035	-0.7083	1.0000
                -3.26449660	1.3120	1.0000
                -1.70936270	0.2236	1.0000
                -2.06451872	0.6392	1.0000
                -2.77457780	1.1390	1.0000""")

    def test_count_lines(self):
        nb_lines = dp.count_lines("foo")
        self.assertEqual(nb_lines, 10, "Number of lines doesn't match. Found %s" % nb_lines)

    def test_parse_data_with_labels(self):
        data_x, data_y = dp.parse_data_with_labels("foo", 2, "\t")

        self.assertListEqual(list(data_x[0]), [-3.60340505, 1.3266], "First line of data doesn't match. Found %s" % data_x[0])
        self.assertListEqual(list(data_y[0]), [1.], "First line of data doesn't match. Found %s" % data_y[0])
        self.assertListEqual(list(data_x[9]), [-2.7745778, 1.139], "Last line of data doesn't match. Found %s" % data_x[9])
        self.assertListEqual(list(data_y[9]), [1.], "Last line of data doesn't match. Found %s" % data_y[9])
