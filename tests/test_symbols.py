import unittest
from hausa_tts.text.symbols import symbols, symbol_to_id, id_to_symbol

class TestSymbols(unittest.TestCase):
    def test_symbols_mapping(self):
        self.assertEqual(len(symbols), len(symbol_to_id))
        self.assertEqual(len(symbols), len(id_to_symbol))
        
    def test_mapping_correctness(self):
        for i, sym in enumerate(symbols):
            self.assertEqual(symbol_to_id[sym], i)
            self.assertEqual(id_to_symbol[i], sym)

if __name__ == '__main__':
    unittest.main()
