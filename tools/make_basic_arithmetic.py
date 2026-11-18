"""Generate the BasicArithmetic evaluation set and its practice pool.

Where it sits in the flow: already run; the two files it wrote are in data/ and are fixed data. It
is seeded, so running it again rewrites the same files. Any Python 3, from the repository root:

    python tools/make_basic_arithmetic.py

The evaluation set: 500 very easy questions, half written with digits and
symbols ("2 + 2 =") and half in words ("two plus two equals"), mostly single-digit numbers with
some two-digit ones. Here that is 250 problems, each written both ways, so the two halves are the
same problems and comparing the forms is a within-problem comparison:
    200 with two single-digit numbers   60 additions, 50 subtractions, 60 multiplications,
                                        30 divisions
     50 with a two-digit number         15 additions (sum at most 99), 15 subtractions,
                                        20 divisions
Subtractions never go below zero. Divisions always come out whole and never divide by zero: with
single-digit numbers
that leaves 32 problems ("6 / 2 ="), and the two-digit ones are the times table read backwards, a
two-digit number divided by a single-digit number with a single-digit result ("56 / 7 =", 58
problems). Rows 0-249 are the symbolic form and rows 250-499 the word form of the same problems in
the same order (`pair` is the problem's index).

How a symbolic question is written varies, because the answer of TinyStories-33M changes with both
the spacing and the multiplication sign: within every operation half the questions are spaced ("2 + 2 =") and
half are not ("2+2="), and 60% of the multiplications use "x" and 40% "*", since "*" may not
occur in such a model's training text. Division is always "/": the division sign certainly does
not. The two choices are made independently, by a seeded draw separate from the one that picks
the problems.

The practice pool (it may overlap the evaluation set, because problems this easy are few): every single-digit problem (100 additions, 55 subtractions, 100
multiplications, 32 divisions) and 213 two-digit ones (78 additions, 77 subtractions and all 58
divisions), each in both forms, 1,000 questions. The two-digit practice additions and subtractions
are drawn independently of the evaluation's.

Writes data/eval/BasicArithmetic_Benchmark.jsonl and data/practice/basic_arithmetic_practice.jsonl,
one JSON object per line: id, pair, source
(the form), problem (the text the model continues), answer, operation, a, b, and for a symbolic
question symbol and spacing ('spaced' or 'tight'; both null in the word form).
"""
import collections
import importlib.util
import json
import os
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Loaded by its path, so this tool needs nothing installed.
_spec = importlib.util.spec_from_file_location('arithmetic', os.path.join(REPO_ROOT, 'wspmathbench', 'arithmetic.py'))
arithmetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arithmetic)

EVAL_FILE = os.path.join(REPO_ROOT, 'data', 'eval', 'BasicArithmetic_Benchmark.jsonl')
PRACTICE_FILE = os.path.join(REPO_ROOT, 'data', 'practice', 'basic_arithmetic_practice.jsonl')
EVAL_SEED, PRACTICE_SEED = 0, 1
EVAL_SINGLE = {'add': 60, 'sub': 50, 'mul': 60, 'div': 30}
EVAL_DOUBLE = {'add': 15, 'sub': 15, 'div': 20}
PRACTICE_DOUBLE = {'add': 78, 'sub': 77, 'div': 58}
LARGEST = 99
STYLE_SEED_OFFSET = 100  # the styles' draw is seeded apart from the problems'
X_SHARE = 0.6  # of the multiplications written with 'x'; the rest use '*'


def single_digit_space(operation):
    """Every problem of this operation with both numbers from 0 to 9. A subtraction's result is
    never negative; a division comes out whole and its divisor is not zero."""
    if operation == 'sub':
        return [(operation, a, b) for a in range(10) for b in range(10) if a >= b]
    if operation == 'div':
        return [(operation, a, b) for a in range(10) for b in range(1, 10) if a % b == 0]
    return [(operation, a, b) for a in range(10) for b in range(10)]


def two_digit_space(operation):
    """Every addition with a two-digit number and a sum up to 99; every subtraction of a number
    from 1 up to a two-digit number; every division of a two-digit number by a single-digit number
    with a single-digit result."""
    if operation == 'add':
        return [(operation, a, b) for a in range(1, LARGEST) for b in range(1, LARGEST)
                if max(a, b) >= 10 and a + b <= LARGEST]
    if operation == 'div':
        return [(operation, b * q, b) for b in range(1, 10) for q in range(1, 10) if b * q >= 10]
    return [(operation, a, b) for a in range(10, LARGEST + 1) for b in range(1, a + 1)]


def draw(rng, single_counts, double_counts):
    """Problems sampled without replacement from each space, then shuffled together."""
    problems = []
    for operation, n in single_counts.items():
        problems += rng.sample(single_digit_space(operation), n)
    for operation, n in double_counts.items():
        problems += rng.sample(two_digit_space(operation), n)
    rng.shuffle(problems)
    return problems


def draw_styles(rng, problems):
    """(symbol, spaced) for each problem's symbolic question. Within each operation half are spaced
    and half tight; multiplications get 'x' with share X_SHARE and '*' otherwise, independently of
    the spacing."""
    styles = [None] * len(problems)
    for operation in arithmetic.OPERATIONS:
        indices = [i for i, problem in enumerate(problems) if problem[0] == operation]
        rng.shuffle(indices)
        spaced = {i: position % 2 == 0 for position, i in enumerate(indices)}
        rng.shuffle(indices)
        symbols = arithmetic.SYMBOLS[operation]
        n_first = round(X_SHARE * len(indices)) if len(symbols) > 1 else len(indices)
        for position, i in enumerate(indices):
            styles[i] = (symbols[0] if position < n_first else symbols[1], spaced[i])
    return styles


def rows(problems, styles, id_prefix):
    """Both forms of every problem: all symbolic rows, then all word rows, in the same order."""
    out = []
    for form in arithmetic.FORMS:
        for pair, ((operation, a, b), (symbol, spaced)) in enumerate(zip(problems, styles)):
            symbolic = form == 'symbolic'
            out.append({'id': f'{id_prefix}-{form}-{pair:04d}', 'pair': pair, 'source': form,
                        'problem': arithmetic.question(operation, a, b, form, symbol, spaced),
                        'answer': str(arithmetic.answer(operation, a, b)), 'operation': operation, 'a': a, 'b': b,
                        'symbol': symbol if symbolic else None,
                        'spacing': ('spaced' if spaced else 'tight') if symbolic else None})
    return out


def write(path, file_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for row in file_rows:
            f.write(json.dumps(row) + '\n')


def describe(label, file_rows):
    answers = collections.Counter(row['answer'] for row in file_rows)
    value, count = answers.most_common(1)[0]
    operations = collections.Counter(row['operation'] for row in file_rows if row['source'] == 'symbolic')
    styles = collections.Counter((row['symbol'], row['spacing']) for row in file_rows if row['source'] == 'symbolic')
    print(f"{label}: symbolic questions by (symbol, spacing) {dict(sorted(styles.items()))}")
    print(f"{label}: {len(file_rows)} questions; problems by operation {dict(operations)}; the most common "
          f"answer is {value} ({count} of {len(file_rows)}, {100 * count / len(file_rows):.1f}%: what always "
          f"answering {value} would score)")


def main():
    evaluation = draw(random.Random(EVAL_SEED), EVAL_SINGLE, EVAL_DOUBLE)
    all_single = {operation: len(single_digit_space(operation)) for operation in arithmetic.OPERATIONS}
    practice = draw(random.Random(PRACTICE_SEED), all_single, PRACTICE_DOUBLE)
    evaluation_rows = rows(evaluation, draw_styles(random.Random(EVAL_SEED + STYLE_SEED_OFFSET), evaluation), 'ba-eval')
    practice_rows = rows(practice, draw_styles(random.Random(PRACTICE_SEED + STYLE_SEED_OFFSET), practice), 'ba-practice')
    write(EVAL_FILE, evaluation_rows)
    write(PRACTICE_FILE, practice_rows)
    describe(f'evaluation -> {EVAL_FILE}', evaluation_rows)
    describe(f'practice -> {PRACTICE_FILE}', practice_rows)
    shared = len(set(evaluation) & set(practice))
    print(f'{shared} of the {len(evaluation)} evaluation problems are also in the practice pool')


if __name__ == '__main__':
    main()
