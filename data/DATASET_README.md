# Dataset README — ET AI Hackathon 2.0 (Problem #8: Industrial Knowledge Intelligence)

## Structure

```
data/
├── corpus/
│   ├── synthetic/      # 11 synthetic PDFs (IR-556, ML-1183, VS-204, INC-2024-07,
│   │                   #   M-118, IR-560, LS-07, M-142, P-210, SH-0717, SOP-09)
│   ├── scanned/        # 3 scanned PDFs (*_SCANNED.pdf)
│   └── real/           # 3 real manuals (PWI_IOM, LKH_manual, MaintMaster_manual)
└── eval/
    ├── benchmark_questions.json
    ├── ground_truth_entities.json
    ├── failuresensoriq_single.json   # ibm-research/FailureSensorIQ single_true_multi_choice_qa
    └── failuresensoriq_multi.json    # ibm-research/FailureSensorIQ multi_true_multi_choice_qa
```

## Sources

| File | Source |
|------|--------|
| `corpus/real/PWI_IOM.pdf` | PumpWorks IOM manual |
| `corpus/real/LKH_manual.pdf` | Alfa Laval LKH centrifugal pump instruction manual |
| `corpus/real/MaintMaster_manual.pdf` | MaintMaster Maintenance Manual 2024 EN |
| `eval/failuresensoriq_*.json` | `ibm-research/FailureSensorIQ` on Hugging Face |

## Status

- `corpus/synthetic/` — **needs manual population** (11 PDFs to be generated/added)
- `corpus/scanned/` — **needs manual population** (3 scanned PDFs to be added)
- `eval/benchmark_questions.json` — **needs manual creation**
- `eval/ground_truth_entities.json` — **needs manual creation**
