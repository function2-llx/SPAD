import pytest

from pumit.numa import _normalize_pci_bus_id, _parse_cpulist


def test_normalize_pci_bus_id() -> None:
    assert _normalize_pci_bus_id('00000000:65:00.0') == '0000:65:00.0'
    assert _normalize_pci_bus_id('00000008:06:00.0') == '0008:06:00.0'
    assert _normalize_pci_bus_id('0008:06:00.0') == '0008:06:00.0'


def test_parse_cpulist() -> None:
    assert _parse_cpulist('0-3,8,10-11') == [0, 1, 2, 3, 8, 10, 11]


def test_parse_empty_cpulist() -> None:
    assert _parse_cpulist('') == []
    assert _parse_cpulist(' \n') == []


def test_parse_cpulist_rejects_empty_segment() -> None:
    with pytest.raises(ValueError):
        _parse_cpulist('0-3,,8')
