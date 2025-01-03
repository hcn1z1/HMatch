import unittest
import numpy as np
import logging
from block_matching import build_pyramid, match_block, compute_residual, process_frame

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestBlockMatching(unittest.TestCase):

    def setUp(self):
        self.image = np.arange(256).reshape(16, 16).astype(np.uint8)
        self.reference = self.image.copy()
        self.block_size = 4
        self.search_window = 4
        self.pyramid_levels = 3
        logging.debug("Set up test case with image shape %s and reference shape %s",
                      self.image.shape, self.reference.shape)
        logging.debug("Block size: %d, Search window: %d, Pyramid levels: %d",
                      self.block_size, self.search_window, self.pyramid_levels)

    def test_build_pyramid(self):
        logging.info("Testing build_pyramid function")
        pyramid = build_pyramid(self.image, self.pyramid_levels)
        logging.debug("Built pyramid with %d levels", len(pyramid))
        for level, layer in enumerate(pyramid):
            logging.debug("Pyramid level %d shape: %s", level, layer.shape)
        self.assertEqual(len(pyramid), self.pyramid_levels, 
                         f"Expected {self.pyramid_levels} pyramid levels, got {len(pyramid)}")
        self.assertEqual(pyramid[1].shape, (8, 8), 
                         f"Expected shape (8,8) at level 1, got {pyramid[1].shape}")
        self.assertEqual(pyramid[2].shape, (4, 4), 
                         f"Expected shape (4,4) at level 2, got {pyramid[2].shape}")

    def test_match_block(self):
        logging.info("Testing match_block function")
        pyramid = build_pyramid(self.reference, self.pyramid_levels)
        target_block = self.image[4:8, 4:8]
        start_coords = (4, 4)
        logging.debug("Target block extracted from coordinates %s:\n%s", start_coords, target_block)
        
        matched_coords, similarity = match_block(
            pyramid,
            target_block,
            start_coords,
            self.search_window,
            self.block_size
        )
        logging.debug("Matched coordinates: %s with similarity score: %s", matched_coords, similarity)
        # Check if the matched coordinates are correct
        self.assertEqual(matched_coords, start_coords, 
                         f"Expected matched coordinates {start_coords}, got {matched_coords}")

    def test_compute_residual(self):
        logging.info("Testing compute_residual function")
        block1 = np.array([[10, 20], [30, 40]])
        block2 = np.array([[5, 15], [25, 35]])
        logging.debug("Block1:\n%s\nBlock2:\n%s", block1, block2)
        residual = compute_residual(block1, block2)
        logging.debug("Computed residual:\n%s", residual)
        expected = np.array([[5, 5], [5, 5]])
        logging.debug("Expected residual:\n%s", expected)
        np.testing.assert_array_equal(residual, expected, 
                                      "Residual does not match the expected output")

    def test_process_frame(self):
        logging.info("Testing process_frame function")
        matched_blocks, residuals = process_frame(
            self.image, self.reference, self.block_size, self.search_window, self.pyramid_levels
        )
        logging.debug("Number of matched blocks: %d", len(matched_blocks))
        logging.debug("Number of residuals: %d", len(residuals))
        logging.debug("Matched blocks coordinates: %s", matched_blocks)
        
        # Verify the number of matched blocks
        expected_blocks = (self.image.shape[0] // self.block_size) * (self.image.shape[1] // self.block_size)
        self.assertEqual(len(matched_blocks), expected_blocks, 
                         f"Expected {expected_blocks} matched blocks, got {len(matched_blocks)}")
        self.assertEqual(len(residuals), expected_blocks, 
                         f"Expected {expected_blocks} residuals, got {len(residuals)}")

if __name__ == "__main__":
    unittest.main()
