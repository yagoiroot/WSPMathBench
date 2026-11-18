"""BasicArithmetic: how a problem is written in each of its two forms, and how an answer is graded.

Where it sits in the flow: used by tools/make_basic_arithmetic.py (writing the questions), by
tools/run_practice_baseline.py (grading a practice baseline) and by llm/benchmarks.py (grading
an evaluation run on BasicArithmetic), so all three agree.

The grading rule is the protocol's: the first number in the model's continuation, written in
digits or in words, is the model's answer. Nothing after it is looked at.

Reads and writes nothing. Standard library only.
"""
import re

FORMS = ('symbolic', 'word')
# operation -> (symbol, words, how the answer is computed). Division is only ever asked where it
# comes out whole and the divisor is not zero (the generator's spaces see to that).
OPERATIONS = {'add': ('+', 'plus', lambda a, b: a + b),
              'sub': ('-', 'minus', lambda a, b: a - b),
              'mul': ('*', 'times', lambda a, b: a * b),
              'div': ('/', 'divided by', lambda a, b: a // b)}
# The signs a symbolic question may use for each operation.
SYMBOLS = {'add': ('+',), 'sub': ('-',), 'mul': ('x', '*'), 'div': ('/',)}

_UNITS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
_TEENS = ('ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
          'nineteen')
_TENS = ('twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')
_UNIT_VALUE = {word: value for value, word in enumerate(_UNITS)}
_TEEN_VALUE = {word: value + 10 for value, word in enumerate(_TEENS)}
_TENS_VALUE = {word: (value + 2) * 10 for value, word in enumerate(_TENS)}
_NUMBER_WORDS = set(_UNIT_VALUE) | set(_TEEN_VALUE) | set(_TENS_VALUE)
_TOKEN = re.compile(r'\d+|[A-Za-z]+')
_JOINERS = (' ', '-')  # what may stand between the words of one number: "twenty three", "twenty-three"


def answer(operation, a, b):
    if operation == 'div' and (b == 0 or a % b):
        raise ValueError(f'{a} / {b} does not come out whole')
    return OPERATIONS[operation][2](a, b)


def number_words(value):
    """0 to 99 in words: 'zero', 'seven', 'fourteen', 'twenty-three'."""
    if not 0 <= value <= 99:
        raise ValueError(f'number_words covers 0 to 99, got {value}')
    if value < 10:
        return _UNITS[value]
    if value < 20:
        return _TEENS[value - 10]
    tens, unit = divmod(value, 10)
    return _TENS[tens - 2] + ('-' + _UNITS[unit] if unit else '')


def question(operation, a, b, form, symbol=None, spaced=True):
    """The text the model continues: '2 + 2 =' or 'two plus two equals'; for a division,
    '6 / 2 =' or 'six divided by two equals'.

    The symbolic form varies in two ways, because a small model's answer depends on both
    (TinyStories-33M gets '1x2=' right and '1 x 2 =' wrong):
        symbol   the operation's sign; None means its default. Multiplication may be 'x' or '*'
                 (SYMBOLS). Division is always '/': the division sign is not in such a model's
                 training text.
        spaced   '2 + 2 =' if true, '2+2=' if false
    Neither applies to the word form.
    """
    default_symbol, word, _ = OPERATIONS[operation]
    if form == 'symbolic':
        symbol = default_symbol if symbol is None else symbol
        if symbol not in SYMBOLS[operation]:
            raise ValueError(f'{operation} is written with one of {SYMBOLS[operation]}, got {symbol!r}')
        return f'{a} {symbol} {b} =' if spaced else f'{a}{symbol}{b}='
    if form == 'word':
        return f'{number_words(a)} {word} {number_words(b)} equals'
    raise ValueError(f'`form` must be one of {FORMS}, got {form!r}')


def first_number(text):
    """The first number in `text`, as an int, or None if it holds none.

    A run of digits is a number ('4', '12'; '4.5' reads as 4), negative if a minus sign stands
    directly before it at the start of a word ('-3'). A number in words is read up to 999
    ('four', 'twenty-three', 'twenty three', 'one hundred and five'). Words of two different
    numbers are not merged across punctuation ('twenty. Three' reads as 20).
    """
    tokens = [(match.group().lower(), match.start(), match.end()) for match in _TOKEN.finditer(text or '')]
    for index, (token, start, _) in enumerate(tokens):
        if token.isdigit():
            negative = start > 0 and text[start - 1] == '-' and (start == 1 or not text[start - 2].isalnum())
            return -int(token) if negative else int(token)
        if token in _NUMBER_WORDS:
            return _read_words(text, tokens, index)
    return None


def _read_words(text, tokens, index):
    """The value of the number phrase that starts at tokens[index] (a number word)."""
    def word_at(i):
        # The next word of the same phrase, or None: it must follow after one space or one hyphen.
        if i < len(tokens) and (i == index or text[tokens[i - 1][2]:tokens[i][1]] in _JOINERS):
            return tokens[i][0]
        return None

    def below_hundred(i):
        word = word_at(i)
        if word in _TEEN_VALUE:
            return _TEEN_VALUE[word], i + 1
        if word in _TENS_VALUE:
            unit = word_at(i + 1)
            if _UNIT_VALUE.get(unit, 0) > 0:
                return _TENS_VALUE[word] + _UNIT_VALUE[unit], i + 2
            return _TENS_VALUE[word], i + 1
        if word in _UNIT_VALUE:
            return _UNIT_VALUE[word], i + 1
        return None, i

    value, after = below_hundred(index)
    if 1 <= value <= 9 and word_at(after) == 'hundred':
        value, after = value * 100, after + 1
        if word_at(after) == 'and':
            after += 1
        rest, _ = below_hundred(after)
        value += rest or 0
    return value
