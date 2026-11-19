# Pre-registration: observed substitution on Trase Brazil soy

Written on 19 November 2026, before any evaluation year was run. The data are Trase's Brazil soy supply chain, version 2.6.1 (`data/external/brazil_soy_v2_6_1_composite.csv`, 2004-2022, CC BY 4.0).

## Task

- **Customer** = importer, **supplier** = exporter. Rows with an unknown importer or exporter are dropped (30% and 3%).
- For a test year Y, the visible graph is every importer-exporter pair seen in years up to Y-1.
- A **query** is an importer c that bought from exporter s in year Y-1 and not in year Y (s is "lost"), where c bought from at least one exporter in Y that it had never used before. The **correct answers** are those new exporters. This is an observed replacement; no similarity proxy is used.
- **Candidates**: exporters with at least one sale in year Y, minus c's visible suppliers and s. The truth is contained in the candidates by construction.
- **Risk** of an exporter = its deforestation exposure per tonne (soy_deforestation_5_year_total_exposure / volume) in the latest year before Y in which it appears, winsorised at the 99th percentile and min-max normalised; exporters without a value get the mean. Exposure is available from 2013.
- **Years**: tuning year 2016; evaluation years 2014, 2015, 2017, 2018, 2020, 2021, 2022. 2019 is excluded: only 9 importers appear in both 2018 and 2019, which signals a break in the identifiers. Each evaluation year is one paired unit, averaged over model seeds 0-2.

## Methods and tuning

Co-supply, popularity, GraphSAGE with binary and softmax losses, RotatE and ComplEx with a softmax loss, and NBFNet (`src/suschain/decision_eval/v5_common.py`). Learned rankers are tuned on the tuning year (mean hit@10; 12 trials for GNNs, 8 for embedding models, a four-point grid for NBFNet at 90 s); "the best learned ranker" is the one with the highest tuning-year hit@10.

## Hypotheses

- **T1** The best learned ranker has a higher hit@10 than co-supply.
- **T2** NBFNet has a higher hit@10 than co-supply.
- **T3 (descriptive pass/fail)** For the best learned ranker and a 20% risk target (alpha = 0.8 x its plain risk@10 in the tuning year), the guaranteed rule has mean test risk <= alpha averaged over 200 random calibration/test splits of the importers, in at least 6 of the 7 evaluation years.

Test: exact two-sided Wilcoxon on the seven paired years (smallest p = 0.0156), Holm over T1-T2. Everything else is exploratory.
