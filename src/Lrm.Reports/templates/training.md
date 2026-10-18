{{! training: filled by Lrm.Reports.Builders.Training from train.* gauges (ReportTemplates contract). Phase 2. }}
# Training: {{value run.title}}

{{value run.banner}}
Final training loss {{value final_loss}} at step {{value last_step}}, after {{value steps}}.

## Loss

{{chart loss}}

{{table stages}}

{{chart lr}}

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
