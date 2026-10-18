{{! inference: filled by Lrm.Reports.Builders.Inference from model.forward and model.layer spans, model.* meters and mem.arena gauges (ReportTemplates contract). }}
# Inference: {{value run.title}}

{{value run.banner}}
| Prefill | Decode | First token | FLOPs per token | Achieved |
| ---: | ---: | ---: | ---: | ---: |
| {{value prefill_rate}} | {{value decode_rate}} | {{value ttft}} | {{value flops_per_token}} | {{value achieved}} |

## Speed

{{chart decode_latency}}

{{chart latency_hist}}

{{table phases}}

## Layers

{{chart layer_time}}

## Memory

{{chart arena_peak}}

{{table arenas}}

## Metering

Resource use attributed to node, component and stage (layers summed). Wall time is span self time, so nested spans are not counted twice.

{{table meters}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
