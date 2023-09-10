# However, due to tokenization inconsistencies, the token boundaries might not perfectly align with the gold spans. Your task is to correct these token boundaries such that they fit within the gold spans.
# token_indices: A list containing sublists of token boundaries. Each sublist represents a sentence and its token boundaries. A token boundary is represented as [start, end], where start and end are character indices (inclusive) of the beginning and end of the token respectively.
# gold_spans: A list containing sublists of gold spans. Each sublist represents a sentence and its gold spans. A gold span is represented as a dictionary {'start': x, 'end': y, 'label': z}, where x and y are the character indices (inclusive) of the beginning and end of the span respectively, and z is the label of the entity.


# Conditions:
# The gold spans' end boundaries must be honored, i.e., no token should end after a gold span's end.
# If a token boundary is adjusted, the adjustment should be minimal, ensuring that the token remains as intact as possible.
# For example:

# Return a list of adjusted token boundaries.


{"start": 0, "end": 9, "label": "Person"}
[[[0, 6], [10, 12]]]

[[[0, 9], [10, 12]]]

{"start": 0, "end": 9, "label": "Person"}
[[[0, 5], [6, 7], [10, 18]]]
[[[0, 5], [6, 9], [10, 18]]]

{"start": 0, "end": 9, "label": "Person"}
[[[0, 5], [6, 11], [12, 13]]]

[[[0, 5], [6, 11], [12, 13]]]


token_indices = [
    [0, 5],
    [6, 7],
    [10, 18],
    [19, 21],
    [22, 32],
    [33, 35],
    [36, 44],
    [45, 50],
    [51, 52],
    [53, 54],
    [58, 60],
    [60, 61],
    [62, 63],
    [64, 67],
    [68, 71],
    [72, 78],
    [79, 82],
    [83, 93],
    [94, 96],
    [97, 105],
    [108, 110],
    [110, 111],
    [112, 114],
    [115, 116],
    [117, 122],
    [123, 127],
    [128, 129],
]


gold_spans = [
    {"start": 0, "end": 9, "label": "Person"},
    {"start": 22, "end": 44, "label": "Misc"},
    {"start": 51, "end": 57, "label": "Organization"},
    {"start": 72, "end": 78, "label": "Event"},
    {"start": 97, "end": 107, "label": "Misc"},
    {"start": 112, "end": 114, "label": "Date"},
    {"start": 117, "end": 127, "label": "Event"},
]


def fix_alignment(token_indices, gold_spans):
    adjusted_token_indices = []

    for sentence_token, sentence_span in zip(token_indices, gold_spans):
        adjusted_tokens = []

        for token in sentence_token:
            for span in sentence_span:

                # if the token ends after the gold span's end, adjust the token's end to the gold span's end
                if token[0] <= span["end"] and token[1] > span["end"]:
                    token[1] = span["end"]

                # break the loop as soon as a match is found to ensure minimal adjustments
                if token[0] >= span["start"] and token[1] <= span["end"]:
                    break

            adjusted_tokens.append(token)

        adjusted_token_indices.append(adjusted_tokens)
    return adjusted_token_indices


token_boundaries_out = fix_alignment(token_indices, gold_spans)
print(token_boundaries_out)

assert token_boundaries_out == [
    [0, 5],
    [6, 9],
    [10, 18],
    [19, 21],
    [22, 32],
    [33, 35],
    [36, 44],
    [45, 50],
    [51, 52],
    [53, 57],
    [58, 60],
    [60, 61],
    [62, 63],
    [64, 67],
    [68, 71],
    [72, 78],
    [79, 82],
    [83, 93],
    [94, 96],
    [97, 107],
    [108, 110],
    [110, 111],
    [112, 114],
    [115, 116],
    [117, 122],
    [123, 127],
    [128, 129],
]
assert token_boundaries_out != token_indices
print("TEST PASSED")
