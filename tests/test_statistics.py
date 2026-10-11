import numpy as np
from attention_lab.statistics import rank_comparison, ranks_descending, paired_changes, bootstrap_indices, from_logits


def test_ties_and_undefined_rank():
    ranks,ties=ranks_descending([.5,.5,.1])
    assert ranks.tolist()==[1.5,1.5,3]
    assert ties==[[0,1]]
    result=rank_comparison([1,1,1],[2,3,4],['a','b','c'])
    assert result['spearman'] is None and result['undefined_reason']=='constant rank vector'
    result=rank_comparison([2,1,3],[1,2,0],['a','b','c'])
    assert abs(result['spearman']+1)<1e-12


def test_signed_paired_changes_and_noop():
    a={'correct':np.array([True,False]),'loss':np.array([0.2,0.4]),'p_correct':np.array([.8,.6])}
    b={'correct':np.array([False,True]),'loss':np.array([0.1,0.3]),'p_correct':np.array([.9,.7])}
    indices=np.array([[0,1],[1,0],[0,0],[1,1]])
    result=paired_changes(a,b,indices)
    assert result['accuracy_drop']['mean']==0
    assert abs(result['loss_increase']['mean']+.1)<1e-12
    assert abs(result['probability_drop']['mean']+.1)<1e-12
    for v in paired_changes(a,a,indices).values():
        assert v=={'mean':0.0,'ci95':[0.0,0.0]}


def test_tied_logits_choose_first_and_extreme_stability():
    values=from_logits([[0,0],[1000,-1000]],np.array([1,1]))
    assert values['prediction'].tolist()==[0,0]
    np.testing.assert_allclose(values['loss'],[np.log(2),2000])
    assert np.array_equal(bootstrap_indices(3,5),bootstrap_indices(3,5))
