| Stage | n | JSON valid | Function-name acc | Exact match | Argument F1 | Call/decline acc | Decline recall | False-decline rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base, zero-shot prompt | 100 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Base, engineered few-shot prompt | 100 | 72.0% | 22.0% | 11.0% | 20.6% | 32.0% | 11.5% | 43.2% |
| SFT (LoRA) | 100 | 98.0% | 69.0% | 34.0% | 69.8% | 69.0% | 0.0% | 4.1% |
| SFT + DPO | 100 | 96.0% | 71.0% | 34.0% | 70.6% | 71.0% | 0.0% | 0.0% |
| Reference: gpt-oss-120b (Groq API), few-shot | 100 | 100.0% | 99.0% | 92.0% | 93.6% | 99.0% | 100.0% | 1.4% |
