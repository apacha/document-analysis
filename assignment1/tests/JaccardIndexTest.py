import unittest
from shapely.geometry import Polygon

from predict import jaccard_index


class JaccardIndexTest(unittest.TestCase):

    def test_perfectly_overlapping_polygons(self):
        # Arrange
        polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        polygon2 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        expected_index = 1

        # Act
        actual_index = polygon1.intersection(polygon2).area / polygon1.union(polygon2).area

        # Assert
        self.assertEqual(expected_index, actual_index)

    def test_non_overlapping_polygons(self):
        # Arrange
        polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        polygon2 = Polygon([(3, 3), (3, 5), (5, 5), (5, 3), (3, 3)])
        expected_index = 0

        # Act
        actual_index = polygon1.intersection(polygon2).area / polygon1.union(polygon2).area

        # Assert
        self.assertEqual(expected_index, actual_index)

    def test_overlapping_polygons(self):
        # Arrange
        polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        polygon2 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])
        expected_index = 1 / 7

        # Act
        actual_index = polygon1.intersection(polygon2).area / polygon1.union(polygon2).area

        # Assert
        self.assertEqual(expected_index, actual_index)

    def test_overlapping_annotations(self):
        # Arrange
        polygon1 = [0, 0, 0, 2, 2, 2, 2, 0]
        polygon2 = [1, 1, 1, 3, 3, 3, 3, 1]
        expected_index = 1 / 7

        # Act
        actual_index = jaccard_index(polygon1, polygon2)

        # Assert
        self.assertEqual(expected_index, actual_index)


if __name__ == '__main__':
    unittest.main()
