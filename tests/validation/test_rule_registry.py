"""Unit tests for RuleRegistry."""

from kiteml.validation.registry import RuleRegistry
from kiteml.validation.rule import ValidationRule


class RegRuleA(ValidationRule):
    rule_id = "R_A"
    name = "Rule A"

    def check(self, df, **kwargs):
        return None


class RegRuleB(ValidationRule):
    rule_id = "R_B"
    name = "Rule B"

    def check(self, df, **kwargs):
        return None


def test_rule_registry_ops():
    registry = RuleRegistry()
    assert len(registry) == 0

    rule_a = registry.register(RegRuleA, tag="dataset")
    assert "R_A" in registry
    assert registry.get("R_A") is rule_a

    registry.register(RegRuleB(), tag="dataset")
    assert len(registry) == 2

    dataset_rules = registry.get_by_tag("dataset")
    assert len(dataset_rules) == 2

    all_rules = registry.list_rules()
    assert len(all_rules) == 2

    registry.unregister("R_A")
    assert "R_A" not in registry
    assert len(registry.get_by_tag("dataset")) == 1

    registry.clear()
    assert len(registry) == 0
