{{! training: filled by Lrm.Reports.Builders.Training from train.* gauges (ReportTemplates contract). Phase 2. }}
# Training: {{value run.title}}

{{value run.banner}}
Final training loss {{value final_loss}} after {{value steps}} steps.

## Loss

{{chart loss}}

{{table stages}}

## Gradients

{{chart grad_heatmap}}

{{chart grad_spread}}

{{chart update_hist}}

## Activations

{{chart act_rms}}

## Run

{{table run.summary}}

{{table run.params}}

### Nodes

{{table run.nodes}}

### Log problems

{{value run.problems}}
