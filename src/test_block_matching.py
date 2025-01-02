import unittest
import numpy as np
from block_matching import build_pyramid, match_block, compute_residual, process_frame

class TestBlockMatching(unittest.TestCase):

    def setUp(self):
        # Set up a dummy image and reference frame for testing
        self.image = np.arange(256).reshape(16, 16).astype(np.uint8)
        self.reference = self.image.copy()
        self.block_size = 4
        self.search_window = 4
        self.pyramid_levels = 3

    def test_build_pyramid(self):
        # Test pyramid construction
        pyramid = build_pyramid(self.image, self.pyramid_levels)
        self.assertEqual(len(pyramid), self.pyramid_levels)
        self.assertEqual(pyramid[1].shape, (8, 8))
        self.assertEqual(pyramid[2].shape, (4, 4))

    def test_match_block(self):
        # Test block matching in a simple scenario
        target_block = self.image[4:8, 4:8]
        start_coords = (4, 4)
        matched_coords, _ = match_block(
            build_pyramid(self.reference, self.pyramid_levels),
            target_block,
            start_coords,
            self.search_window,
            self.block_size
        )
        # Check if the matched coordinates are correct
        self.assertEqual(matched_coords, start_coords)

    def test_compute_residual(self):
        # Test residual calculation
        block1 = np.array([[10, 20], [30, 40]])
        block2 = np.array([[5, 15], [25, 35]])
        residual = compute_residual(block1, block2)
        expected = np.array([[5, 5], [5, 5]])
        np.testing.assert_array_equal(residual, expected)

    def test_process_frame(self):
        # Test processing an entire frame
        matched_blocks, residuals = process_frame(
            self.image, self.reference, self.block_size, self.search_window, self.pyramid_levels
        )
        # Verify the number of matched blocks
        expected_blocks = (self.image.shape[0] // self.block_size) * (self.image.shape[1] // self.block_size)
        self.assertEqual(len(matched_blocks), expected_blocks)
        self.assertEqual(len(residuals), expected_blocks)

if __name__ == "__main__":
    unittest.main()
