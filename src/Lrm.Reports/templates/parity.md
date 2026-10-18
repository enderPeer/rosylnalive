{{! parity: filled by Lrm.Reports.Builders.Parity from golden.compare checks (ReportTemplates contract). }}
# Parity: {{value run.title}}

{{value run.banner}}
{{value verdict}}

- Comparisons: {{value checks}}, failed: {{value failed}}
- Pairs: {{value pairs}}; cases: {{value cases}}
- Closest to its bound: {{value worst}}

## Error by layer

Every per-layer tap's rel_l2 against the golden, summarized per layer over all cases and taps. A pair passes when every dot and whisker stays under its bound line.

{{chart rel_l2_by_layer}}

{{chart ratio_histogram}}

## Per pair

{{#each pairs}}
### {{value name}}: {{value verdict}}

{{value checks}} comparisons, {{value failed}} failed, worst ratio {{value worst}} of the bound.

{{chart heatmap}}

{{/each}}
## Per case

{{table cases}}

## {{value closest.title}}

{{table closest}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
