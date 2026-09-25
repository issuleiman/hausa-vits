import re
import unicodedata

_abbreviations = {
    "dr.": "dakta",
    "prof.": "farfesa",
    "malam": "malam",
    "mal.": "malam",
    "hajiya": "hajiya",
    "haj.": "hajiya",
    "alhaji": "alhaji",
    "alh.": "alhaji"
}

_number_words = {
    0: "sifili", 1: "ɗaya", 2: "biyu", 3: "uku", 4: "huɗu", 5: "biyar",
    6: "shida", 7: "bakwai", 8: "takwas", 9: "tara", 10: "goma",
    11: "goma sha ɗaya", 12: "goma sha biyu", 13: "goma sha uku",
    14: "goma sha huɗu", 15: "goma sha biyar", 16: "goma sha shida",
    17: "goma sha bakwai", 18: "goma sha takwas", 19: "goma sha tara",
    20: "ashirin", 30: "talatin", 40: "arba'in", 50: "hamsin",
    60: "sittin", 70: "saba'in", 80: "tamanin", 90: "tis'in",
    100: "ɗari", 1000: "dubu"
}

def expand_number(num: int) -> str:
    if num in _number_words:
        return _number_words[num]
    
    if num < 100:
        tens = (num // 10) * 10
        ones = num % 10
        return f"{_number_words[tens]} da {_number_words[ones]}"
    
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        hundred_str = f"ɗari {_number_words[hundreds]}" if hundreds > 1 else "ɗari"
        if remainder == 0:
            return hundred_str
        return f"{hundred_str} da {expand_number(remainder)}"
        
    if num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        thousand_str = f"dubu {expand_number(thousands)}" if thousands > 1 else "dubu"
        if remainder == 0:
            return thousand_str
        return f"{thousand_str} da {expand_number(remainder)}"
        
    return str(num)

def expand_numbers_in_text(text: str) -> str:
    def replace_num(match):
        num_str = match.group(0)
        # Handle decimal
        if '.' in num_str:
            parts = num_str.split('.')
            whole = expand_number(int(parts[0]))
            fraction = " ɗigo " + " ".join([expand_number(int(digit)) for digit in parts[1]])
            return whole + fraction
        # Handle negative
        if num_str.startswith('-'):
            return "kasa da " + expand_number(int(num_str[1:]))
            
        return expand_number(int(num_str))
        
    return re.sub(r'-?\d+(\.\d+)?', replace_num, text)

def is_ajami(text: str) -> bool:
    """Detect if text is in Ajami (Arabic script)."""
    # Simple check for Arabic unicode block
    return any('\u0600' <= c <= '\u06FF' for c in text)

def transliterate_ajami_to_boko(text: str) -> str:
    """Basic Ajami to Boko transliteration (placeholder for full implementation)."""
    # Requires a complex mapping rule set. For now, return as is if not implemented fully.
    if is_ajami(text):
        pass # full ajami translit logic goes here
    return text

def expand_abbreviations(text: str) -> str:
    for abbr, expanded in _abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', expanded, text, flags=re.IGNORECASE)
    return text

def hausa_cleaners(text: str) -> str:
    """Pipeline for Hausa text normalization."""
    # 1. Ajami to Boko
    text = transliterate_ajami_to_boko(text)
    
    # 2. Unicode normalization (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # 3. Lowercase conversion
    text = text.lower()
    
    # 4. Abbreviation expansion
    text = expand_abbreviations(text)
    
    # 5. Number expansion
    text = expand_numbers_in_text(text)
    
    # 6. Punctuation normalization
    text = re.sub(r'[^\w\s\.,?!\-…]', '', text)
    
    # 7. Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
