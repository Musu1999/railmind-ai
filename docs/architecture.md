# RailMind AI — design notes

## Design principles
1. **Governance-native, not bolted on.** Every service inherits `TabularService.infer()`, which enforces the data-quality gate, logs telemetry and produces an explanation. There is no ungoverned prediction path.
2. **Agents are accountable.** The planner records every tool call; a separate ValidatorAgent re-derives facts (train runs that day, class exists, fare matches) before a plan is shown; booking requires a one-time approval token.
3. **Autonomy within policy.** The supervisor acts on its own only where `config.AUTO_REMEDIATION` allows. Actions that change business decisions (for example fraud thresholds) or fairness questions go to a human approval queue.
4. **Verify, don't assume.** A remediation is only "resolved" after fresh traffic passes the same checks that raised the alert.
5. **Offline-first GenAI.** The support assistant answers from a curated knowledge base and live tools; an LLM may only rephrase, and the guard blocks anything not grounded.

## Data model (synthetic)
| Dataset | Rows | Purpose |
|---|---|---|
| trains | 162 | 54 corridors × directions; type, classes, capacity, run days, punctuality |
| bookings | 60 000 | WL/RAC bookings with final outcome after chart preparation |
| delays | 50 000 | Train runs with weather, congestion, maintenance blocks and arrival delay |
| station_load | 60 480 | Station footfall by hour |
| fares | 40 000 | Flexi/Tatkal fare quotes vs days-before and occupancy |
| demand | 85 224 | Daily bookings per corridor (seasonality, festivals, trend) |
| users / choices | 5 000 / 52 116 | Passenger profiles and historical recommendation sessions |
| sessions | 25 000 | Booking sessions with labelled attack types |
| pnrs | 400 | PNRs for support-assistant lookups |

The generator embeds known causal structure (festival intensity suppresses cancellations, fog dominates northern winter delays, flexi fares step up 10 % per 10 % sold, and so on) so models learn something real and injected drift is meaningful.

## Model inventory
| Model | Algorithm | Primary metric |
|---|---|---|
| confirmation_predictor | HistGradientBoosting (binary) + multi-class status model | Brier score (calibration) |
| delay_predictor | HGB regression + P10/P90 quantile models + congestion regressor | MAE |
| fare_predictor | HGB regression + occupancy curve + demand forecaster + Isolation Forest | MAE |
| fraud_detector | Class-weighted HGB + Isolation Forest + rules | F1 (recall/FPR monitored) |
| train_recommender | Pointwise learning-to-rank (HGB) | Hit@1, NDCG@3 |
| support_assistant | TF-IDF (char + word) RAG + tools + guard | Grounding, quality, CSAT |
| travel_planner_agent | Rule-based NLU + 7 sub-agents | Validity rate, anomaly rate |

## Governance thresholds (config.py)
| Signal | Threshold |
|---|---|
| PSI watch / drift | 0.10 / 0.25 |
| Classification metric drop | 0.05 absolute |
| Regression error increase | 35 % relative |
| Fraud recall floor / FPR ceiling | 0.85 / 0.05 |
| Group error-rate gap | 0.10 |
| GenAI grounding minimum | 0.35 |
| Data-quality block | score < 0.70 or any hard-constraint breach |
| p95 latency SLA | 250 ms |
| Agent tool-call budget | 25 per run |

## Production deployment path
```
Clients ─► API gateway (OAuth, rate limits, WAF)
            ├─► Passenger services (K8s, autoscaled)  ─► model server (KServe/Triton)
            │        └─► feature store (Feast) ◄─ Kafka ◄─ PRS · NTES · IMD · payment gateway
            └─► Governance services
                     telemetry: Kafka → ClickHouse/Prometheus
                     registry: MLflow · audit: WORM object storage
                     supervisor: scheduled + event-driven (alert webhooks)
                     approvals: integrated with ticketing/on-call
```
The Python interfaces (`Store`, `registry`, `simulator`) are the seams where these managed services plug in.
