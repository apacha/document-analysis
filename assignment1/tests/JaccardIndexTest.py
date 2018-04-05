import unittest


class JaccardIndexTest(unittest.TestCase):

    def test_perfectly_overlapping_polygons(self):
        # Arrange
        expected_index = 1.0

        # Act
        actual_index = 0.93

        # Assert
        self.assertEqual(expected_index, actual_index)


if __name__ == '__main__':
    unittest.main()
