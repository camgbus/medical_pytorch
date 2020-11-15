from mp.data.datasets.ds_mr_heart_decathlon import DecathlonLeftAtrium

def test_ds_label_merging():
    data = DecathlonLeftAtrium()
    assert data.label_names == ['background', 'left atrium']
    assert data.nr_labels == 2
    assert data.modality == 'MRI'
    assert data.size == 20
    assert data.name == 'DecathlonLeftAtrium'