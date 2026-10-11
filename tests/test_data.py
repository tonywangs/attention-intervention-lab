import pytest
from attention_lab.data import generate, oracle, belongs


def test_data_disjoint_balanced_and_oracle():
    sets = []
    for split in ("train", "dev", "eval"):
        rows = generate(split, 512)
        assert rows == generate(split, 512)
        contexts = set()
        counts = [0]*4
        for row in rows:
            tokens = row["tokens"]
            mapping = {tokens[i]: tokens[i+1]-16 for i in range(0, 8, 2)}
            assert row["answer"] == mapping[tokens[-1]]
            assert oracle(tokens) == (row["answer"], row["value_position"])
            assert belongs(tokens, split)
            contexts.add(tuple(sorted(mapping.items())))
            counts[row["value_position"]//2] += 1
        assert counts == [128]*4
        assert len(contexts) == len(rows)
        sets.append(contexts)
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])


def test_oracle_rejects_duplicates_and_bad_queries():
    tokens = generate("dev", 1)[0]["tokens"]
    tokens[2] = tokens[0]
    with pytest.raises(ValueError):
        oracle(tokens)
    with pytest.raises(ValueError):
        oracle([1]*10)
