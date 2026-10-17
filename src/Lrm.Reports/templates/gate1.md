{{! gate1: filled by Lrm.Reports.Builders.Gate1 from gate1.check checks and logits golden.compare checks (ReportTemplates contract). }}
# Gate 1: {{value run.title}}

{{value run.banner}}
{{value verdict}}

## Matrix

{{table matrix}}

{{chart by_group}}

## Logits per gate prompt

{{chart logits}}

## Every check

{{table checks}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
