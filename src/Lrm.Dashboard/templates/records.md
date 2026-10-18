{{! records: the dashboard's view of a run whose kind has no report template, filled by Lrm.Dashboard.Records from every record of the run (ADR T2-0001). }}
# {{value run.kind}}: {{value run.title}}

{{value run.banner}}
{{value verdict}}

{{value summary}}

## Checks

{{chart checks}}

{{table checks}}

## Records

Every record name of the run with what it adds up to. Spans and GPU intervals are summed by duration, counters and meters by value, gauges show their last value and range.

{{chart spans}}

{{table records}}

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
