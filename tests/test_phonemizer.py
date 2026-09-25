import unittest
from hausa_tts.text.phonemizer import HausaPhonemizer

class TestPhonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = HausaPhonemizer()
        
    def test_basic_word(self):
        phones, tones = self.phonemizer.phonemize("sannu")
        self.assertIsInstance(phones, list)
        self.assertIsInstance(tones, list)
        self.assertEqual(len(phones), len(tones))

if __name__ == '__main__':
    unittest.main()
