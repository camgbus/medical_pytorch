from mp.data.datasets.ds_mr_lung_decathlon_reg import DecathlonLung

def test_ds_label():
    data = DecathlonLung()
    assert data.modality == 'CT'
    assert data.size == 63
    assert data.name == 'DecathlonLung'