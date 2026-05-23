"""Pure pre-flight validator for ComfyUI workflow JSONs.

Given a workflow (API format, keyed by node-id) and a ComfyUI /object_info
slice (keyed by class_type), returns a list of human-readable failure strings.
Empty list = the workflow passes pre-flight.

Three checks per node:
  1. class_type is registered in the running ComfyUI
  2. every REQUIRED input the node declares is provided by the workflow
  3. every literal input value that targets an enum field is in the enum list

Connection-typed inputs (lists shaped [node_id, output_idx]) are runtime-
resolved values from upstream nodes; they satisfy "provided" and skip the
enum check (we can't validate them at this layer without topology).

No I/O. Easily unit-testable. Used by automation/smoke_preset.py.
"""

from __future__ import annotations


def _is_connection(value: object) -> bool:
    """True if `value` looks like a ComfyUI connection: [node_id_str, output_idx_int]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _enum_options(spec: object) -> list[str] | None:
    """If `spec` is an enum field spec, return its option list; else None.

    ComfyUI enum specs look like `[["opt1", "opt2"], {...}]` — the first element
    is a list of strings. Primitive specs are `["INT", {...}]` (string), and
    connection specs are `["LATENT"]` (string). Only the list-of-strings shape
    is an enum.
    """
    if not isinstance(spec, list) or not spec:
        return None
    head = spec[0]
    if isinstance(head, list) and all(isinstance(o, str) for o in head):
        return head
    return None


def validate(
    workflow: dict,
    object_info: dict,
    skip_enum_fields: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Validate `workflow` against `object_info`.

    Args:
        workflow: ComfyUI workflow in API format — `{node_id: {class_type, inputs}}`.
        object_info: ComfyUI's /object_info slice — `{class_type: {input: {required, optional}, ...}}`.
        skip_enum_fields: Optional set of `(node_id, field_name)` pairs whose
            enum check should be skipped. Use this for fields that will be
            populated at submit time by an out-of-band mechanism (e.g. smoke
            gate's `--input` upload to LoadImage / VHS_LoadVideo). The dropdown
            options at object_info-query time don't include the not-yet-uploaded
            file, so a literal enum check would false-positive. Required and
            class-installed checks still run.

    Returns:
        List of failure strings (empty list = the workflow is valid).
    """
    skip = skip_enum_fields or set()
    failures: list[str] = []
    for node_id, node in workflow.items():
        ct = node.get("class_type")
        if ct not in object_info:
            failures.append(f"Node {node_id} ({ct}): class not installed")
            continue

        required = object_info[ct].get("input", {}).get("required", {})
        provided = node.get("inputs", {})

        for field_name, spec in required.items():
            if field_name not in provided:
                failures.append(
                    f"Node {node_id} ({ct}): missing required input '{field_name}'"
                )
                continue

            value = provided[field_name]
            if _is_connection(value):
                continue

            if (node_id, field_name) in skip:
                continue

            options = _enum_options(spec)
            if options is not None and value not in options:
                failures.append(
                    f"Node {node_id} ({ct}): '{value}' not in {options} for input '{field_name}'"
                )

    return failures
