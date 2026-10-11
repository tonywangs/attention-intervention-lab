from attention_lab.artifacts import Budget
from attention_lab.data import generate
from attention_lab.model import Config
from attention_lab.train import train_one


def test_exhausted_budget_preserves_learning_floor_checkpoint(tmp_path):
    result=train_one(Config(),generate('train',8),generate('dev',8),123,
                     {'steps':100,'batch':128,'lr':.001,'validate_every':100},tmp_path,Budget(1e-10))
    assert result['timed_out'] is True
    assert result['learning_gate'] is False
    assert result['selected_step']==0
    assert result['history']==[]
    assert (tmp_path/'seed-123.npz').is_file()
    assert (tmp_path/'seed-123.json').is_file()
