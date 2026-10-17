{{! kernels: filled by Lrm.Reports.Builders.Kernels from conformance.case checks and gpu.dispatch intervals (ReportTemplates contract). }}
# Kernels: {{value run.title}}

{{value run.banner}}
{{value verdict}} {{value budget}}

- Conformance cases: {{value cases}}, failed: {{value failed}}
- GPU dispatches: {{value dispatches}}, total GPU time {{value gpu_total}}

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

{{table matrix}}

### Failing cases

{{table failures}}

## GPU time

{{chart gpu_by_kernel}}

{{chart layer_stack}}

## Timeline

{{chart timeline}}

{{chart timeline_zoom}}

{{chart dispatch_hist}}

### Slowest dispatches

{{table slowest}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
