## Code Design Principles

### 1. Performance and Resource Efficiency

- Avoid unnecessary loops, repeated traversals, deep copies, and intermediate data structures.
- Prefer direct data access and lightweight transformations over redundant processing.
- Use deep copies only when independent mutable state is genuinely required.
- Do not introduce caching, batching, or other performance optimizations without a clear benefit.

### 2. Avoid Over-Defensive Programming

- Trust established upstream contracts and validated inputs.
- Do not repeatedly validate inputs or add fallback branches within internal methods when their preconditions are guaranteed by upstream logic.
- Avoid speculative exception handling, redundant null checks, and silent fallback values that conceal programming errors.
- Validate data at system boundaries; keep internal implementations focused on their explicit responsibilities.
- Add defensive handling only for realistic failure scenarios, not hypothetical misuse.

### 3. Clean and Maintainable Code

- Keep implementations simple, readable, and free of redundant logic, dead code, and unnecessary abstractions.
- Prefer straightforward control flow over excessive branching and deeply nested conditions.
- Centralize genuinely reusable utility functions in the appropriate `utils` directory.
- Search for and reuse existing utilities before introducing new ones.
- Do not create utility wrappers that merely delegate to an existing function without adding meaningful functionality.
- Remove obsolete code and unnecessary compatibility layers when they are no longer required.

### 4. Abstraction and Module Boundaries

- Place shared functionality at the lowest common architectural layer accessible to all intended consumers.
- When reusing functionality currently defined inside a specific module, evaluate whether it should be promoted to a shared module or an appropriate `utils` directory.
- Avoid cross-module dependencies on private implementation details.
- Keep domain-specific logic within its owning module; extract only functionality with genuine reuse potential.
- Preserve clear dependency direction and avoid circular dependencies.

## Protocol Design Principles

1. **Protocols are independent.** Keep native field mappings and protocol algorithms within their owning codecs. A protocol implementation must not import another protocol's implementation. Protocols communicate through the middleware contract; a codec may recognize its own source context for native replay.
2. **Shared base classes stay protocol-neutral.** Use them to define contracts, assemble the pipeline, and enforce common invariants. Keep protocol-specific branches, field rules, and event handling out of base classes; delegate differences to protocol-owned implementations. Shared helpers require genuinely shared semantics or an explicitly documented internal trade-off.
3. **Each codec owns its protocol differences.** Keep native schemas, defaults, field mappings, identities, stream handling, and compatibility rules inside the corresponding protocol's codec package. Share only logic with the same semantics across all consumers.
4. **Encoding and decoding have distinct loss boundaries.** Native protocol -> encoder -> middleware projects only fields supported by the middleware and silently discards unsupported native fields. Middleware -> decoder -> target protocol records semantic loss when a supported middleware value cannot be represented by the target protocol or its configured capabilities. Do not report discarded native-only fields as decoder losses or silently drop supported middleware semantics.
5. **Fix root causes instead of adding patches.** Correct the owning contract, state, or data flow. Do not conceal broken invariants with one-off protocol exceptions, guessed values, duplicated conversions, downstream reconstruction of missing upstream state, or silent fallbacks. Remove obsolete workarounds when fixing their cause.

### Accepted Internal Trade-offs

- Protocol guessing may coordinate recognition of different protocols when the caller does not specify one. This is input identification, not a conversion dependency between protocols.
- The existing internal grouping of consecutive Responses assistant items and their shared reasoning-feedback decision is intentional. Keep this convention internal; do not add protocol-specific compatibility controls to public middleware models to remove it.
- Business contracts cover `MyLLM.generate` and `MyCodec.encode` / `decode` / `convert`. Internal components rely on pipeline preconditions. Do not add recovery or duplicate validation to guarantee behavior after callers deliberately corrupt returned snapshots or event sequences.

## Documentation Requirements

1. **Organize documentation into four parts:** usage, development, testing, and review. Keep [docs/README.md](docs/README.md) as the shared navigation entry. Existing file paths referenced by tests may remain; classify pages by audience and purpose rather than duplicating their contents.
2. **Usage documentation serves package users.** Explain how to use the supported public interfaces, their arguments, return values, configuration, lifecycle, errors, and examples. Business entry points are `MyLLM.generate` and the translation methods on `MyCodec`; describe registration as an extension mechanism, not another generation API. Do not present internal pipeline methods as independently supported business interfaces.
3. **Development documentation serves project developers.** Explain current design choices, their intent and trade-offs, responsibility boundaries, data flow, and extension contracts. Verify descriptions against the current implementation; distinguish accepted decisions from future proposals.
4. **Testing documentation contains reproduction guides and result summaries, including performance tests.** Separate commands and acceptance criteria from dated results. Record the tested baseline, environment, scope, outcomes, and limitations when evidence is available; identify missing evidence. Distinguish local SDK simulations from live provider calls and do not present historical results as current guarantees.
5. **Review documentation contains code reviews and fix records.** Record the affected baseline, root cause, impact, evidence, and final disposition. Preserve accepted no-fix decisions and their business scope; do not turn them back into pending fixes without new evidence or a change in requirements.
6. **Maintain one authoritative explanation per topic.** Link across the four parts instead of copying long instructions, result tables, or implementation descriptions. Update local links when moving pages. Keep examples aligned with current interfaces and preserve protocol snapshot provenance. Never put credentials or raw sensitive reasoning/signature data in documentation or reports.
