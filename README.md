# Multilingual Sycophancy Experiment

A configurable pipeline for validating vignette variants, generating model
responses, parsing final choices, scoring misconception endorsement, and
exporting screening diagnostics.

## Current workbook status

The supplied workbook has 60 candidate scenarios and five English condition
variants in `Prompt_Variants`. The candidates are marked as pending review and
the sheet does not contain Hindi prompt text or an approval field. The default
configuration therefore excludes these rows from model runs. Add a
`review_status` column with `approved` (or another configured approved status)
after review, and add language-specific rows before requesting those languages.
Do not enable `dataset.include_unreviewed` for a substantive experiment.

The loader supports either one row per language/condition or language-suffixed
text fields such as `context_en`, `context_hi`, `option_A_en`, and
`option_A_hi`. A shared `scenario_id` links translated versions.

## Setup

```powershell
python -m pip install -r requirements.txt
python run.py validate
```

Configure reviewed languages, conditions, and models in
`config/experiment.yaml`. Supported API families are `openai_compatible` and
`anthropic`; credentials must be supplied through environment variables.
Example model entries:

```yaml
models:
  openai_model:
    provider: openai_compatible
    model: provider-model-id
    api_key_env: OPENAI_API_KEY
    endpoint: https://api.openai.com/v1
  anthropic_model:
    provider: anthropic
    model: provider-model-id
    api_key_env: ANTHROPIC_API_KEY
    endpoint: https://api.anthropic.com/v1
```

Then run and analyze:

```powershell
$env:OPENAI_API_KEY = "..."
python run.py run
python run.py analyze
```

Generation records are appended to `data/results/generations.csv`. Successful
job IDs are skipped on resume; failed attempts remain in the file and are
retried by a later run. A job ID changes if its prompt, condition, model ID, or
experiment identity changes. Model calls are not made by `validate` or
`analyze`.

## Outputs

- `generations.csv`: prompt, raw response, metadata, status, and errors.
- `scores.csv`: parsed choice, correctness, endorsement, uncertainty, refusal,
  parsing, and contradiction flags.
- `item_metrics.csv`: response counts and rates by vignette, language, model,
  and condition.
- `vignette_metrics.csv`: neutral-to-strong shift and directional mirroring.
- `vignette_summary.csv`: separate shortlist diagnostics and candidate ranks.
- `reports/pilot/*.png` and matching CSVs: plots and exact data used to create
  each figure.

Endorsement rates use valid A/B choices as their denominator. UNSURE,
refusals, invalid outputs, and parse failures have separate counts/rates. The
primary shift is `strong_misconception - neutral`. Directional mirroring is
the change in misconception choice between strong-misconception and
strong-correct stances.

Shortlist ranking uses mean shift and leaves diagnostics visible rather than
combining them into a weighted score. Its repeatability flag measures the
fraction of available language/model item contrasts that are positive; it is
not a repeated-generation statistical test. Re-run shortlisted items with
fresh generations before interpreting their effect as confirmatory.

## Tests

```powershell
pytest
```
