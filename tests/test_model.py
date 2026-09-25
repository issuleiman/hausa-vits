import unittest
import torch
import yaml
import os
from hausa_tts.model.vits import HausaVITS

class TestModel(unittest.TestCase):
    def setUp(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'base.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = HausaVITS(self.config)
        
    def test_model_instantiation(self):
        self.assertIsNotNone(self.model)

if __name__ == '__main__':
    unittest.main()
