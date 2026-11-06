# Resume bullets: tool-call fine-tuning, SFT → DPO (all numbers measured in this repo)

Scope note for interviews: the measured run is small-scale (SmolLM2-135M-Instruct, 240 training examples,
2-core CPU, 100 held-out test examples). The Qwen3-8B QLoRA configs for A100/T4 are written but were not run.

- Built an end-to-end tool-calling fine-tuning pipeline (data → prompt baseline → LoRA SFT → scorer-labelled DPO →
  evaluation → report) with Hugging Face TRL + PEFT. Cleaned glaive-function-calling-v2 from 112,960 raw chats to a
  validated 4,000-example dataset (3,000/400/600 splits, 20% "no tool applies" declines) with a data card and 21
  pytest tests.
- Fine-tuned SmolLM2-135M-Instruct with LoRA (4.9M trainable params, 3.5% of the model) on 240 examples. Against
  the best engineered few-shot prompt on 100 held-out examples: exact-match tool calls went from 11% to 34% (3.1×),
  JSON validity from 72% to 98%, argument F1 from 20.6% to 69.8%, and the false-decline rate fell from 43.2% to 4.1%.
- Added DPO on top of the SFT adapter, with 29 preference pairs labelled automatically by the task scorer from 4
  samples per prompt. Reported the outcome as measured: false declines went from 3 to 0, but exact match stayed at
  34% and JSON validity dipped from 98% to 96%. Traced the flat result to the pairs themselves: only 2 of 29 had a
  decline as the chosen answer, while the SFT model's main failure was 0% decline recall.
- Benchmarked against gpt-oss-120b (Groq API, 92% exact match on the same 100 examples). An error audit showed 7 of
  its 8 misses came from dataset label quirks rather than model mistakes. Made every stage resumable (eval caches,
  trainer checkpoints, a supervisor loop), so the run survived OOM kills and two sandbox reboots on a 7.8 GB CPU box.
