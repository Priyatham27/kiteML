"""
registry.py — Registry for managing and looking up KiteML validation rules.
"""

from typing import TypeVar

from kiteml.validation.rule import ValidationRule

R = TypeVar("R", bound=ValidationRule)


class RuleRegistry:
    """
    Central registry for storing, looking up, and instantiating validation rules.
    """

    def __init__(self) -> None:
        self._rules: dict[str, ValidationRule] = {}
        self._tags: dict[str, set[str]] = {}  # tag -> set of rule_ids

    def register(
        self,
        rule: ValidationRule | type[ValidationRule],
        tag: str | None = None,
    ) -> ValidationRule:
        """
        Register a rule instance or rule class.

        Parameters
        ----------
        rule : ValidationRule or type[ValidationRule]
            Rule object or class to register.
        tag : str, optional
            Category tag to associate with the rule.

        Returns
        -------
        ValidationRule
            The instantiated and registered rule object.
        """
        if isinstance(rule, type) and issubclass(rule, ValidationRule):
            instance = rule()
        elif isinstance(rule, ValidationRule):
            instance = rule
        else:
            raise TypeError(f"Expected ValidationRule or subclass, got {type(rule)}")

        rule_id = instance.rule_id
        self._rules[rule_id] = instance

        if tag:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(rule_id)

        return instance

    def get(self, rule_id: str) -> ValidationRule | None:
        """Retrieve a rule by its rule_id."""
        return self._rules.get(rule_id)

    def get_by_tag(self, tag: str) -> list[ValidationRule]:
        """Retrieve all rules registered under a given tag."""
        rule_ids = self._tags.get(tag, set())
        return [self._rules[rid] for rid in rule_ids if rid in self._rules]

    def list_rules(self) -> list[ValidationRule]:
        """List all registered rules."""
        return list(self._rules.values())

    def unregister(self, rule_id: str) -> None:
        """Unregister a rule by rule_id."""
        if rule_id in self._rules:
            del self._rules[rule_id]
        for rule_set in self._tags.values():
            rule_set.discard(rule_id)

    def clear(self) -> None:
        """Clear all registered rules."""
        self._rules.clear()
        self._tags.clear()

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self._rules


# Global default registry instance
global_rule_registry = RuleRegistry()
