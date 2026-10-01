import copy

import pytest

from climate_rag.scifact_natural_selection import SALT, digest, select_metadata


def rows(n=40):
    return [{"id": i + 1, "component": f"c{i // 2:03}"} for i in range(n)]


def test_whole_component_exclusion_and_no_replacement():
    pool = rows()
    value = select_metadata(pool, [pool[0]], [pool[2]], ["protected"])
    assert value["counts"]["excluded_pool_claims"] == 4
    assert value["counts"]["eligible_components"] == 18
    assert value["counts"]["shortfall"] == 6
    assert not {r["component"] for r in value["selected"]} & {"c000", "c001", "protected"}
    assert len({r["component"] for r in value["selected"]}) == 18
    assert value["caps"]["physical_generations"] == 18 * 7
    assert not value["replacement_allowed"]


def test_fixed_hash_order_not_original_cohort_order():
    pool = rows(144)
    value = select_metadata(pool, [], [], [])
    assert value == select_metadata(list(reversed(pool)), [], [], [])
    components = sorted({r["component"] for r in pool}, key=lambda c: (digest([SALT, "component", c]), c))[:24]
    assert [r["component"] for r in value["selected"]] == components
    for chosen in value["selected"]:
        same = [r["id"] for r in pool if r["component"] == chosen["component"]]
        assert chosen["id"] == min(same, key=lambda i: (digest([SALT, "claim", chosen["component"], i]), i))
    assert value["caps"]["physical_generations"] == 168
    assert value["caps"]["rerank_pairs"] == 960


@pytest.mark.parametrize("key", ["role", "label", "status", "training_ready", "cohort", "claim", "result"])
def test_selector_rejects_outcome_or_text_fields(key):
    pool = rows()
    pool[0][key] = "do_not_consume"
    with pytest.raises(ValueError, match="projection_only"):
        select_metadata(pool, [], [], [])


def test_mapping_drift_duplicates_and_protected_fail_closed():
    pool = rows()
    bad = copy.deepcopy(pool[0])
    bad["component"] = "changed"
    with pytest.raises(ValueError, match="drift"):
        select_metadata(pool, [bad], [], [])
    with pytest.raises(ValueError, match="duplicate"):
        select_metadata(pool + [pool[0]], [], [], [])
    with pytest.raises(ValueError, match="protected"):
        select_metadata(pool, [], [], [pool[0]["component"]])


def test_empty_eligible_pool_is_explicit_not_replenished():
    pool = rows(2)
    value = select_metadata(pool, [pool[0]], [], [])
    assert value["selected"] == []
    assert value["counts"]["shortfall"] == 24
    assert value["caps"]["physical_generations"] == 0
