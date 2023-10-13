from typing import TypeVar, Generic

from python_util.logger.logger import LoggerFacade

T = TypeVar("T")


def has_sub_patterns(subpattern, pattern):
    len_p = len(pattern)
    len_s = len(subpattern)

    if len_s > len_p:
        return False

    for i in range(len_p - len_s + 1):
        if pattern[i:i + len_s] == subpattern:
            return True

    return False


def create_rolling_patterns(subpattern):
    all_patterns = []
    for i in range(len(subpattern)):
        patterns = []
        for j in range(len(subpattern)):
            patterns.append(subpattern[(i + j) % len(subpattern)])
        all_patterns.append(patterns)

    return all_patterns


def has_subpattern(subpattern, pattern):
    return any([has_sub_patterns(sub_pattern_item, pattern)
                for sub_pattern_item in create_rolling_patterns(subpattern)])


def is_repetitions(sub_pattern: list[int], pattern: list[int]):
    sub_pattern_str = ','.join([str(p) for p in sub_pattern])
    pattern_str = ','.join([str(p) for p in pattern])

    out = pattern_str.replace(sub_pattern_str, '|___|').split('|___|')
    if len(out) != 0:
        for out_pattern in out:
            if out_pattern != ',' and out_pattern != '' and \
                    not has_subpattern([int(i) for i in out_pattern.split(',') if len(i) != 0], sub_pattern):
                return False

    if len(out) > 0:
        pattern_str = pattern_str.replace(out[0], '', 1)
    if len(out) > 1:
        if out[-1] != ',' and out[-1] != '':
            reversed_pattern_str = ''.join(reversed([i for i in pattern_str]))
            out_ = out[-1].strip().replace(',', '')
            reversed_value_removed = reversed_pattern_str.replace(out_, '', 1)
            pattern_str = ''.join(reversed([j for j in reversed_value_removed]))

    pattern = [int(i) for i in pattern_str.split(',') if i != ',' and i != '']

    for i, num in enumerate(sub_pattern):
        for j, pattern_start in enumerate(pattern):
            if pattern_start == num:
                for q in range(len(pattern)):
                    sub_pattern_item = sub_pattern[(q + i) % (len(sub_pattern))]
                    pattern_item = pattern[(q + j) % (len(pattern))]
                    if sub_pattern_item != pattern_item:
                        return False
                return True

    return True


def is_repetition(sub_pattern: list[int], pattern: list[int]):
    for sub_pattern_item in create_rolling_patterns(sub_pattern):
        if is_repetitions(sub_pattern_item, pattern):
            return True

    return False


class PatternKeepingList(Generic[T]):
    # TODO:
    def __init__(self, max_length, do_combinatorial_patterns: bool = False):
        self.do_combinatorial_patterns = do_combinatorial_patterns
        self.max_length = max_length
        self.data = []
        self.pattern = None
        self.restart_idx = -1
        self.potential_patterns = {}

    def add_str(self, element):
        if self.restart_idx == -1:
            self.add(int(element))

    def add(self, element):
        if (self.pattern is not None and self.do_combinatorial_patterns) or self.restart_idx != -1:
            return  # Pattern has been found, no more additions

        if self.max_length >= len(self.data):
            self.data.append(element)

        # Initialize potential patterns when enough data is accumulated
        if self.do_combinatorial_patterns:
            self.potential_patterns[len(self.data)] = [i for i in range(len(self.data))]

        # Update potential patterns
        if 2 < len(self.data) <= self.max_length and self.do_combinatorial_patterns:
            self.update_potential_patterns()

        # Determine pattern when maximum length is reached
        if self.max_length == len(self.data):
            self.find_pattern()

    def update_potential_patterns(self):
        new_potential_patterns = {}
        for pattern_length, start_indices in self.potential_patterns.items():
            for start_index in start_indices:
                length_ = self.data[start_index: start_index + pattern_length]
                pattern_length_ = self.data[
                                  start_index + pattern_length:min(start_index + pattern_length + pattern_length,
                                                                   len(self.data))]
                if length_ == pattern_length_:
                    if pattern_length + 1 not in new_potential_patterns:
                        new_potential_patterns[pattern_length + 1] = []
                    new_potential_patterns[pattern_length + 1].append(start_index)
        for key, pattern in new_potential_patterns.items():
            self.potential_patterns[key] = pattern

    def detect_pattern_of_len(self, patt):
        return len([i for i in range(len(self.data)) if self.data[i: i + len(patt)] == patt])


    def find_repeating_sequences(self, sequence_in, n_sequences_check: int):
        """Finds all repeating sequences within a sequence.

        Args:
          sequence: The sequence to search.

        Returns:
          A list of all repeating sequences.
          :param n_sequences_check:
        """
        sequences = []
        count = 0
        sequence = sequence_in
        while len(sequence) > 1 and count < n_sequences_check:
            left, right = sequence[:len(sequence) // 2], sequence[len(sequence) // 2:]
            if left == right:
                sequences.append(left)
            sequence = left if len(left) < len(right) else right
            count += 1
        return sequences if len(sequence) == 0 and len(sequence[0]) != 1 else [sequence_in]

    def find_pattern(self):
        if len(self.data) != 1 and self.data[-1:][0] == self.data[0]:
            s = self.find_repeating_sequences(self.data[:-1], 10)
            self.pattern = s[0]
            self.restart_idx = len(self.data)
        elif self.restart_idx != -1:
            return self.pattern
        elif self.do_combinatorial_patterns:
            self.find_combinatorial_pattern()
        else:
            self.pattern = self.data


    def find_combinatorial_pattern(self):
        sorted_patterns = sorted(self.potential_patterns.items(), key=lambda item: item[0])

        for pattern_length, start_indices in sorted_patterns:
            for start_index in start_indices:
                potential_pattern = self.data[start_index: start_index + pattern_length]
                if self.detect_pattern_of_len(potential_pattern) > 1:
                    if not self.pattern or len(potential_pattern) > len(self.pattern):
                        self.pattern = potential_pattern

        if not self.pattern:
            return

        patterns = [self.pattern] if self.pattern else None

        for length, pattern_start in self.potential_patterns.items():
            for start_index in pattern_start:
                pattern_to_check = self.data[start_index: start_index + length]
                for i, pattern in enumerate(patterns):
                    if pattern_to_check not in patterns and is_repetition(pattern_to_check, pattern):
                        patterns.append(pattern_to_check)

        sorted_val = sorted(patterns, key=lambda x: len(x))
        biggest = len(sorted_val) - 1
        for i, pattern in enumerate(sorted_val):
            while True:
                biggest_ = sorted_val[biggest]
                if is_repetition(pattern, biggest_):
                    if len(biggest_) == len(pattern):
                        self.pattern = pattern
                        return
                    biggest = biggest - 1
                else:
                    break
                if biggest == i:
                    self.pattern = pattern
                    return

    def get_pattern_str(self):
        self.find_pattern()
        return [str(i) for i in self.get_pattern()]

    def get_pattern(self):
        if self.pattern and len(self.pattern) != 0 and all(
                pattern_item == self.pattern[0] for pattern_item in self.pattern):
            return [self.pattern[0]]
        elif not self.pattern:
            return self.data
        else:
            return self.pattern
