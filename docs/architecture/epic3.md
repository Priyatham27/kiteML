# Epic 3 Architecture: Developer Experience (DX) Framework

The DX Framework intercepts runtime exceptions, attaches diagnostic suggestions, manages escalation warning policies, and powers the 14-command CLI ecosystem.

```text
User API / CLI --> DXPipeline --> Exception Manager (KML-XXX)
                              --> Warning Manager (KML-W-XXX)
                              --> Suggestion Manager (Fuzzy Typo Matcher)
                              --> Diagnostics Box Output
```
