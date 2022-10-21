import pandas as pd
import numpy as np
import pytest
from experiment.analyze import analyze_v1_11_tools as tools

def _make_long_choice_rows(*, chid, nchoices, choice_idx, id=0):
    assert choice_idx in range(nchoices)
    return [
        dict(chid=chid, alt=i, choice=choice_idx == i, id=id)
        for i in range(nchoices)
    ]

def needs_rlang():
    try:
        tools.setup_r()
    except Exception as e:
        pytest.skip(f'could not init solway: {e}')
        return

def test_nullmodel_same_number_choices():
    needs_rlang()
    rdf = tools.convert_df_to_rdf(pd.DataFrame([
        *_make_long_choice_rows(chid=0, nchoices=3, choice_idx=0),
        *_make_long_choice_rows(chid=1, nchoices=3, choice_idx=2),
        *_make_long_choice_rows(chid=2, nchoices=3, choice_idx=2),
    ]))
    mlogitdf = tools.ro.r.convert_rdf_to_dfidx(rdf)
    expected = np.log(np.array([1/3, 1/3, 1/3])).sum()
    assert np.isclose(
        expected,
        tools.ro.r.logLik(tools.ro.r.nullmodel(mlogitdf))[0],
    )
    # Should also match our old version that only works for same # options
    assert np.isclose(
        expected,
        tools.ro.r.logLik(tools.ro.r.nullmodel_same_number_options(mlogitdf))[0],
    )

def test_nullmodel_diff_number_choices():
    needs_rlang()
    rdf = tools.convert_df_to_rdf(pd.DataFrame([
        *_make_long_choice_rows(chid=0, nchoices=2, choice_idx=0),
        *_make_long_choice_rows(chid=1, nchoices=3, choice_idx=0),
        *_make_long_choice_rows(chid=2, nchoices=4, choice_idx=0),
        *_make_long_choice_rows(chid=3, nchoices=6, choice_idx=0),
    ]))
    mlogitdf = tools.ro.r.convert_rdf_to_dfidx(rdf)
    expected = np.log(np.array([1/2, 1/3, 1/4, 1/6])).sum()
    assert np.isclose(
        expected,
        tools.ro.r.logLik(tools.ro.r.nullmodel(mlogitdf))[0],
    )
