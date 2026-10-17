{{! thinking: filled by Lrm.Reports.Builders.Thinking from roslyn.call spans and think.step events (ReportTemplates contract). Phase 3. }}
# Thinking: {{value run.title}}

{{value run.banner}}
{{value calls}} Roslyn calls from {{value layers_calling}} layers, {{value ok_rate}} of them ok; {{value steps}} thinking steps.

## Trace

{{chart trace}}

{{chart by_api}}

{{chart latency}}

## By API

{{table apis}}

## First calls

{{table calls}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
