#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lemmo import LEMMORecognitionRuntime, LEMMODescriptionRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LEMMO free-generation request")
    parser.add_argument("--model", choices=("LEMMO_Recognition", "LEMMO_Description"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--signal", required=True, help=".npy or .csv IQ file")
    parser.add_argument("--sample-rate", type=float, required=True, help="sample rate in Hz")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--input-setting",
        help="acquisition-setting text; required for LEMMO_Description",
    )
    parser.add_argument("--task", choices=("amc", "interference", "uav", "wtc"))
    parser.add_argument("--system-prompt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    if args.model == "LEMMO_Recognition":
        if args.task is None:
            parser.error("--task is required for LEMMO_Recognition")
        runtime = LEMMORecognitionRuntime(args.config, args.device)
        kwargs = {}
        if args.system_prompt is not None:
            kwargs["system_prompt"] = args.system_prompt
        answer = runtime.answer(
            args.signal,
            args.sample_rate,
            args.task,
            args.question,
            args.max_new_tokens or 64,
            **kwargs,
        )
    else:
        if args.input_setting is None:
            parser.error("--input-setting is required for LEMMO_Description")
        runtime = LEMMODescriptionRuntime(args.config, args.device)
        kwargs = {}
        if args.system_prompt is not None:
            kwargs["system_prompt"] = args.system_prompt
        answer = runtime.answer(
            args.signal,
            args.sample_rate,
            args.input_setting,
            args.question,
            args.max_new_tokens or 512,
            **kwargs,
        )
    print(json.dumps({"model": args.model, "answer": answer}, ensure_ascii=False))


if __name__ == "__main__":
    main()

