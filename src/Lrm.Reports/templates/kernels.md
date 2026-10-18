{{! kernels: filled by Lrm.Reports.Builders.Kernels from conformance.case checks, gpu.dispatch intervals and gpu.fp32_denormals events (ReportTemplates contract). }}
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

## fp32 denormals

The mode each shader backend measured with the `scale_f32` denormal probe right after loading its kernels (GpuBackendInterface). It is fixed within an execution class, and it decides what bit-exact means for a kernel that does fp32 arithmetic (KernelAbiAndCatalog, kernel rules).

{{table denormals}}

The probe's words (`scale_f32`, a = 0.5): each input, what preserve and flush give, and what each measurement returned.

{{table denormal_words}}

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
