from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping


@dataclass
class ModelArgumens:
    model_name_or_path: Optional[str] = field(
        default="llm",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    dataset_dir: Optional[str] = field(
        default="chinese llama alpaca", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )


parser = HfArgumentParser((ModelArgumens, DataTrainingArguments))
a_args, b_args = parser.parse_args_into_dataclasses()


def main():
    print(a_args.model_name_or_path)
    print(b_args.dataset_dir)


if __name__ == "__main__":
    main()
