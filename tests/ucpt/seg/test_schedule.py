from pumit.ucpt.seg.schedule import neck_da_schedule


def test_da0_upsamples_depth_both_stages():
    sched = neck_da_schedule(0)
    assert [s.upsample_depth for s in sched] == [True, True]


def test_da3_upsamples_depth_once():
    sched = neck_da_schedule(3)
    assert [s.upsample_depth for s in sched] == [True, False]


def test_da4_no_depth_upsample():
    sched = neck_da_schedule(4)
    assert [s.upsample_depth for s in sched] == [False, False]


def test_da_value_climbs_after_depth_freeze():
    sched = neck_da_schedule(3)
    assert [s.da for s in sched] == [3, 4]


def test_none_is_2d():
    sched = neck_da_schedule(None)
    assert all(s.da is None for s in sched)
    assert all(s.upsample_depth is False for s in sched)
