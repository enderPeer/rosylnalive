{{! cluster: filled by Lrm.Reports.Builders.Cluster from cluster.unit and cluster.step spans and cluster.* events (ReportTemplates contract). }}
# Cluster: {{value run.title}}

{{value run.banner}}
{{value verdict}}

| Work units | Reissued | Nodes lost | Divergences | Replica digests equal |
| ---: | ---: | ---: | ---: | ---: |
| {{value units}} | {{value reissued}} | {{value lost}} | {{value divergences}} | {{value digests}} |

## Pipeline

Green units were accepted, grey ones are matching duplicates from the other replica, amber ones were reissued after a node loss, and red ones were cut off when their node died.

{{chart gantt}}

{{chart step_latency}}

## Nodes

{{chart busy}}

{{table nodes}}

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
