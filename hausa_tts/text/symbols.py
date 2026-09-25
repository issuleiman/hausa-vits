_pad = '_'
_punctuation = ' .,?!-…'
_vowels = 'a e i o u aː eː iː oː uː'
_diphthongs = 'ai au'
_consonants = 'b t d k g ɓ ɗ ƙ dʒ tʃ tsʼ f s z ʃ h m n ɲ ŋ l r ɽ w j ʔ kʷ gʷ kʲ gʲ ʔʲ'

# Split into lists
_pad_list = [_pad]
_punct_list = list(_punctuation)
_vowel_list = _vowels.split()
_diphthong_list = _diphthongs.split()
_consonant_list = _consonants.split()

# Main symbol inventory
symbols = _pad_list + _punct_list + _vowel_list + _diphthong_list + _consonant_list

# Dictionaries
SYMBOL_TO_ID = {s: i for i, s in enumerate(symbols)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(symbols)}
PAD_ID = 0
n_symbols = len(symbols)

# Tone markers
tone_symbols = ['<TONE_PAD>', '<H>', '<L>', '<F>']
TONE_TO_ID = {t: i for i, t in enumerate(tone_symbols)}
ID_TO_TONE = {i: t for i, t in enumerate(tone_symbols)}
n_tones = len(tone_symbols)
