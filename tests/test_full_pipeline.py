"""Integration test verifying full FacePipeline with MediaPipe FaceLandmarker."""

import os
import unittest
import numpy as np
import cv2

from src.config import AppConfig
from src.core.face_pipeline import FacePipeline
from src.schema import EligibilityLevel, FaceRole


class TestFacePipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AppConfig()
        cls.pipeline = FacePipeline(cls.config)
        cls.pipeline.initialize()

    @classmethod
    def tearDownClass(cls):
        cls.pipeline.close()

    def test_pipeline_on_synthetic_face(self):
        # Create a synthetic image with a stylized face schema (eyes, nose, mouth)
        # to verify that the pipeline processes it without crashing and handles zero-face or detection gracefully
        h, w = 480, 640
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw face oval
        cv2.ellipse(canvas, (320, 240), (120, 160), 0, 0, 360, (220, 180, 150), -1)
        # Draw eyes
        cv2.circle(canvas, (270, 200), 20, (255, 255, 255), -1)
        cv2.circle(canvas, (370, 200), 20, (255, 255, 255), -1)
        cv2.circle(canvas, (270, 200), 8, (50, 40, 30), -1)
        cv2.circle(canvas, (370, 200), 8, (50, 40, 30), -1)
        # Draw mouth
        cv2.ellipse(canvas, (320, 300), (40, 20), 0, 0, 180, (50, 20, 20), -1)

        # Process frame 1
        results1 = self.pipeline.process_frame(canvas, timestamp=0.033)
        # Should execute successfully without throwing errors
        self.assertIsInstance(results1, list)

        # Process frame 2
        results2 = self.pipeline.process_frame(canvas, timestamp=0.066)
        self.assertIsInstance(results2, list)


if __name__ == "__main__":
    unittest.main()
