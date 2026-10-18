{{! goldens: filled by Lrm.Reports.Builders.Goldens from the goldens.* records of tools/goldens (ReportTemplates contract). }}
# Golden set: {{value run.title}}

{{value run.banner}}
{{value verdict}}

- Sets: {{value sets}}
- Regeneration: {{value regen}}
- Double run: {{value double_run}}
- Checks: {{value checks}}

## Size

{{chart budget}}

{{table sets}}

{{chart files}}

{{table files}}

## Checks

Every `goldens.*` check of the run, and the oracle self-tests that carry a measured value.

{{table checks}}

{{#each corpus}}
## Tokenizer corpus

{{value summary}}

{{chart categories}}

{{table kinds}}

{{/each}}
## Phases

{{table phases}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
