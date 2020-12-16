import mp.utils.helper_functions as hf

def test_date_time():
    date = hf.get_time_string(True)
    assert len(date) == 21
    date = hf.get_time_string(False)
    assert len(date) == 19

def test_avg_dicts():
    d1 = {'last': {'A': 1.0, 'B': 0.5, 'C': 0.3}, 'best': {'D': 0.74}, 'another': 75}
    d2 = {'last': {'A': 0.0, 'B': 0.5, 'C': 0.3}, 'best': {'D': 0.25}}
    avg, std = hf.average_dictionaries(d1, d2)
    expected_result = {'best': {'D': 0.495}, 'last': {'A': 0.5, 'B': 0.5, 'C': 0.3}}
    assert avg == expected_result
