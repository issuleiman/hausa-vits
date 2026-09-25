import re
from typing import List, Tuple

class HausaPhonemizer:
    def __init__(self, use_tone_marks: bool = True):
        """Initialize the Hausa G2P converter.
        Args:
            use_tone_marks: If True, parse tone diacritics from input text.
                           If False, output TONE_PAD for all phonemes.
        """
        self.use_tone_marks = use_tone_marks
        
        self.vowel_mapping = {
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
            'aa': 'aː', 'ee': 'eː', 'ii': 'iː', 'oo': 'oː', 'uu': 'uː',
            'ā': 'aː', 'ē': 'eː', 'ī': 'iː', 'ō': 'oː', 'ū': 'uː',
            'ai': 'ai', 'au': 'au'
        }
        
        self.tone_mapping = {
            'á': ('a', '<H>'), 'é': ('e', '<H>'), 'í': ('i', '<H>'), 'ó': ('o', '<H>'), 'ú': ('u', '<H>'),
            'à': ('a', '<L>'), 'è': ('e', '<L>'), 'ì': ('i', '<L>'), 'ò': ('o', '<L>'), 'ù': ('u', '<L>'),
            'â': ('a', '<F>'), 'ê': ('e', '<F>'), 'î': ('i', '<F>'), 'ô': ('o', '<F>'), 'û': ('u', '<F>')
        }
        
        # Longest match first
        self.digraphs = {
            'sh': 'ʃ', 'ts': 'tsʼ', 'fy': 'fʲ', 'ky': 'kʲ', 'gy': 'gʲ',
            'kw': 'kʷ', 'gw': 'gʷ', 'ch': 'tʃ', "'y": 'ʔʲ', 'ƴ': 'ʔʲ',
            "b'": "ɓ", "d'": "ɗ", "k'": "ƙ"
        }
        
        self.single_chars = {
            'ɓ': 'ɓ', 'ɗ': 'ɗ', 'ƙ': 'ƙ', 'c': 'tʃ', 'j': 'dʒ', 'y': 'j', 'r': 'r'
        }

    def phonemize(self, text: str) -> Tuple[List[str], List[str]]:
        """Convert Hausa text to phoneme and tone sequences."""
        phonemes = []
        tones = []
        
        # Simple word tokenization keeping punctuation
        words = re.findall(r'[\w\u00C0-\u017F]+|[.,?!\-… ]', text)
        
        for word in words:
            if word in ' .,?!-…':
                phonemes.append(word)
                tones.append('<TONE_PAD>')
            else:
                word_ph, word_tn = self._grapheme_to_phoneme(word)
                phonemes.extend(word_ph)
                tones.extend(word_tn)
                
        return phonemes, tones

    def _grapheme_to_phoneme(self, word: str) -> Tuple[List[str], List[str]]:
        """Convert a single Hausa word to phonemes and tones."""
        ph_list = []
        tn_list = []
        
        # Word initial glottal stop for vowels
        if len(word) > 0 and (word[0].lower() in 'aeiouáéíóúàèìòùâêîôûāēīōū'):
            ph_list.append('ʔ')
            tn_list.append('<TONE_PAD>')
            
        i = 0
        word = word.lower()
        while i < len(word):
            # Check for tone marked vowels
            if self.use_tone_marks and word[i] in self.tone_mapping:
                base_vowel, tone = self.tone_mapping[word[i]]
                ph_list.append(base_vowel)
                tn_list.append(tone)
                i += 1
                continue
            elif not self.use_tone_marks and word[i] in self.tone_mapping:
                base_vowel, _ = self.tone_mapping[word[i]]
                ph_list.append(base_vowel)
                tn_list.append('<TONE_PAD>')
                i += 1
                continue
                
            # Check digraphs and trigraphs
            matched = False
            for length in [2, 1]:
                if i + length <= len(word):
                    chunk = word[i:i+length]
                    if chunk in self.vowel_mapping:
                        ph_list.append(self.vowel_mapping[chunk])
                        tn_list.append('<H>' if self.use_tone_marks else '<TONE_PAD>')
                        i += length
                        matched = True
                        break
                    elif chunk in self.digraphs:
                        ph = self.digraphs[chunk]
                        if ph == 'fʲ':
                            ph_list.extend(['f', 'j'])
                            tn_list.extend(['<TONE_PAD>', '<TONE_PAD>'])
                        else:
                            ph_list.append(ph)
                            tn_list.append('<TONE_PAD>')
                        i += length
                        matched = True
                        break
                    elif chunk in self.single_chars:
                        ph_list.append(self.single_chars[chunk])
                        tn_list.append('<TONE_PAD>')
                        i += length
                        matched = True
                        break
            
            if not matched:
                # Handle 'n' before velar
                if word[i] == 'n' and i + 1 < len(word) and word[i+1] in ['k', 'g']:
                    ph_list.append('ŋ')
                else:
                    ph_list.append(word[i])
                tn_list.append('<TONE_PAD>')
                i += 1
                
        return ph_list, tn_list

    def _syllabify(self, phonemes: List[str]) -> List[List[str]]:
        """Split phoneme sequence into syllables following Hausa CV/CVC/CVV patterns."""
        syllables = []
        # Complex implementation depending on phonotactics
        # Placeholder for basic syllable splitting logic
        return syllables
