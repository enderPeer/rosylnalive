{{! cluster: filled by Lrm.Reports.Builders.Cluster from C1's cluster.* records (ClusterProtocol "Telemetry"; ReportTemplates contract). }}
# Cluster: {{value run.title}}

{{value run.banner}}
{{value verdict}}

| Outcome | Work units | Cross-checked | Reissued | Nodes lost | Divergences | Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| {{value status}} | {{value units}} | {{value cross_checked}} | {{value reissued}} | {{value lost}} | {{value divergences}} | {{value epoch}} |

Run digest: {{value digest}}

## Pipeline

Each bar is a work unit the coordinator accepted, drawn on the node whose result it took, from the assignment to the acceptance. Green units were accepted from their first assignment; amber ones were reissued to another node after the first missed its deadline. Markers are node losses, kills, restarts and demotions.

{{chart gantt}}

{{chart step_latency}}

{{chart deadlines}}

{{chart latency}}

## Nodes

{{chart busy}}

{{table nodes}}

### Counters

{{table counters}}

### Cluster events

{{table events}}

## Metering

{{table meters}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
