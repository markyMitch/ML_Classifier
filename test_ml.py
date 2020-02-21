import unittest
import ml

class MyTestCase(unittest.TestCase):

    def test_shorten_year(self):
        self.assertEqual(14, ml.shorten_year(2014))


if __name__ == '__main__':
    unittest.main()
