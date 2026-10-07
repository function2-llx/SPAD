def test_auxseghead_not_exported():
    import pumit.ucpt.seg as seg_pkg
    assert not hasattr(seg_pkg, 'AuxSegHead'), 'AuxSegHead must be removed from seg exports'
