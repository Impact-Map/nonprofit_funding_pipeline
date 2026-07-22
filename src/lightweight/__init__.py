"""Phase 1 lightweight pipeline (USAspending-only, business-types-driven).

See `Methodology_FederalAssistance_501c3_Lightweight_FY22-FY25.docx`.

This subpackage replaces the recipient_match + classify steps of the full
pipeline with simpler implementations that rely solely on USAspending data:

  - recipient_filter: applies the M-with-exclusions rule per Section 3.2.
  - categorize:       heuristic carve-outs for International/Hospital/
                      Educational/Core (Section 4).
  - tables:           analytic table assembly (Section 11 schema, simplified).
"""
