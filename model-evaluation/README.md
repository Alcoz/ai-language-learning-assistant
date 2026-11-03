# Model evaluation

Evaluates the models of the assistant. Each type of model has its own folder.

- `stt/`: runs the Whisper models on the labeled recordings of the bucket, with `make evaluate-stt`. Start the MLflow server first, with `make mlflow-run`.
- `llm/`: scores the teacher's answers traced in Langfuse, with `make evaluate-llm`. A local Ollama model (`EVALUATION_OLLAMA_MODEL`) judges each answer from 1 to 5, and the score is attached to its generation with the judge's reasoning as comment:
  - `answer_coherence`: the answer fits the student's last message. Skipped on the opening message;
  - `vocabulary_level`: the vocabulary fits the requested CEFR level;
  - `sentence_complexity`: the sentence structure fits the requested CEFR level.

Each run scores every generation again and overwrites its previous scores. Generations traced before the teaching profile was added to their metadata are skipped.

`make benchmark-llm` compares teacher models (`TEACHER_MODELS` in `llm/benchmark_llm.py`). It copies the conversation of every traced answer into the `teacher-conversations` Langfuse dataset, then works in two phases, so that Ollama keeps a single model in memory at a time:

1. each model answers every conversation, in one experiment per model, named after it and the launch time, in `Teacher benchmark`. Each answer gets `generation_seconds` and `tokens_per_second`, measured by Ollama without the time spent loading the model;
2. the judge scores every answer on the three criteria above. Compare the runs in Langfuse, under Datasets > `teacher-conversations`.
