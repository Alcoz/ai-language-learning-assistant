from dotenv import load_dotenv

load_dotenv()

import json
import os
from datetime import datetime
from typing import Any

from conversation_engine import TEACHING_PROFILE
from langfuse import Evaluation, Langfuse
from langfuse.experiment import ExperimentItemResult
from ollama import Client
from tqdm import tqdm

from model_evaluation.llm.evaluate_llm import (
    DEFAULT_JUDGE_MODEL,
    format_conversation,
    get_applicable_criteria,
    judge_answer,
    list_generations,
)

DATASET_NAME = "teacher-conversations"
EXPERIMENT_NAME = "Teacher benchmark"
TEACHER_MODELS = [
    "qwen3.5:4b",
    "qwen3.5:latest",
    "gemma3:4b",
    "gemma3:12b",
    "llama3.2:3b",
]

ollama_client = Client()
judge_model = os.environ.get("EVALUATION_OLLAMA_MODEL", DEFAULT_JUDGE_MODEL)


def copy_generations_to_dataset(langfuse: Langfuse) -> None:
    """Copy the conversation of every traced teacher answer into the benchmark dataset."""
    langfuse.create_dataset(name=DATASET_NAME)

    for generation in list_generations(langfuse):
        metadata = generation.metadata or {}
        profile = {
            key: metadata.get(key, value) for key, value in TEACHING_PROFILE.items()
        }
        langfuse.create_dataset_item(
            id=generation.id,
            dataset_name=DATASET_NAME,
            input=json.loads(generation.input),
            metadata=profile,
            source_trace_id=generation.trace_id,
            source_observation_id=generation.id,
        )


def make_teacher_task(model: str, progress_bar: tqdm):
    """Build the task that makes the given model answer a conversation of the dataset."""

    def answer_conversation(*, item: Any, **kwargs) -> dict[str, Any]:
        response = ollama_client.chat(model=model, messages=item.input, think=False)
        generation_seconds = (response.total_duration - response.load_duration) / 1e9
        progress_bar.update()

        return {
            "answer": response.message.content,
            "generation_seconds": generation_seconds,
            "tokens_per_second": response.eval_count / (response.eval_duration / 1e9),
        }

    return answer_conversation


def score_speed(*, output: dict[str, Any], **kwargs) -> list[Evaluation]:
    """Turn the speed measured by Ollama into scores."""
    return [
        Evaluation(name="generation_seconds", value=output["generation_seconds"]),
        Evaluation(name="tokens_per_second", value=output["tokens_per_second"]),
    ]


def answer_with_every_model(
    langfuse: Langfuse, models: list[str], items: list[Any]
) -> list[ExperimentItemResult]:
    """First phase: make each model answer every conversation, one model loaded at a time."""
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    item_results = []

    for model in models:
        with tqdm(total=len(items), desc=model, unit="answer") as progress_bar:
            result = langfuse.run_experiment(
                name=EXPERIMENT_NAME,
                run_name=f"{model} {started_at}",
                data=items,
                task=make_teacher_task(model, progress_bar),
                evaluators=[score_speed],
                max_concurrency=1,
                metadata={"teacher_model": model, "judge_model": judge_model},
            )
        item_results.extend(result.item_results)

    return item_results


def judge_every_answer(
    langfuse: Langfuse, item_results: list[ExperimentItemResult]
) -> None:
    """Second phase: score every answer with the judge, which stays loaded all along."""
    for item_result in tqdm(item_results, desc=f"judge {judge_model}", unit="answer"):
        if item_result.output is None:
            continue

        messages = item_result.item.input
        conversation = format_conversation(messages, item_result.output["answer"])

        for criteria_name, criteria_description in get_applicable_criteria(
            messages
        ).items():
            verdict = judge_answer(
                ollama_client,
                judge_model,
                criteria_description,
                item_result.item.metadata,
                conversation,
            )
            if verdict is None:
                continue

            langfuse.create_score(
                trace_id=item_result.trace_id,
                name=criteria_name,
                value=verdict.score,
                data_type="NUMERIC",
                comment=verdict.reasoning,
                metadata={"judge_model": judge_model},
            )


def main() -> None:
    """Make each teacher model answer the traced conversations, and compare them in Langfuse."""
    langfuse = Langfuse()
    copy_generations_to_dataset(langfuse)
    dataset = langfuse.get_dataset(DATASET_NAME)

    item_results = answer_with_every_model(langfuse, TEACHER_MODELS, dataset.items)
    judge_every_answer(langfuse, item_results)

    langfuse.flush()


if __name__ == "__main__":
    main()
