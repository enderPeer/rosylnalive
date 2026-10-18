{{! routing: filled by Lrm.Reports.Builders.Routing from route.usage counters and route.entropy gauges (ReportTemplates contract). Phase 3. }}
# Routing: {{value run.title}}

{{value run.banner}}
{{value experts}} experts over {{value layers}} layers; the charts show step {{value step}} unless they say otherwise.

## Expert usage

{{chart usage_heatmap}}

{{chart expert_mix}}

## Router entropy

{{chart entropy_by_layer}}

{{chart entropy_over_steps}}

## Top expert per layer

{{table top}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
