from pydantic import BaseModel
from .schema import SyntheticData, SyntheticDataList
import os
import time
import asyncio
import logging
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.models import DeepEvalBaseLLM
from deepteam.utils import sanitize_prompt_for_encoding

MAX_RETRIES = os.getenv("DEEPTEAM_MAX_RETRIES", 3)


def generate(
    prompt: str,
    schema: BaseModel,
    model: DeepEvalBaseLLM = None,
) -> BaseModel:
    """
    Generate schema using the provided model with retry logic.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        The validated schema object
    """
    prompt = sanitize_prompt_for_encoding(prompt)
    _, using_native_model = initialize_model(model=model)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            if using_native_model:
                res, _ = model.generate(prompt=prompt, schema=schema)
                if res is None:
                    raise ValueError("Model returned None.")
                return res
            else:
                try:
                    res = model.generate(prompt=prompt, schema=schema)
                    if res is None:
                        raise ValueError("Model returned None.")

                    if isinstance(res, str):
                        data = trimAndLoadJson(res)
                        return schema(**data)
                    else:
                        return res
                except TypeError:
                    res = model.generate(prompt)
                    if res is None:
                        raise ValueError("Model returned None.")

                    data = trimAndLoadJson(res)
                    if schema == SyntheticDataList:
                        data_list = [SyntheticData(**item) for item in data["data"]]
                        return SyntheticDataList(data=data_list)
                    else:
                        return schema(**data)

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2**attempt
                logging.warning(
                    f"Generation failed on attempt {attempt + 1}. Retrying in {sleep_time}s... Error: {e}"
                )
                time.sleep(sleep_time)

    raise RuntimeError(
        f"Failed to generate after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


async def a_generate(
    prompt: str,
    schema: BaseModel,
    model: "DeepEvalBaseLLM" = None,
) -> BaseModel:
    """
    Asynchronously generate schema using the provided model with retry logic.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        The validated schema object
    """
    prompt = sanitize_prompt_for_encoding(prompt)
    _, using_native_model = initialize_model(model=model)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            if using_native_model:
                res, _ = await model.a_generate(prompt=prompt, schema=schema)
                if res is None:
                    raise ValueError("Model returned None.")
                return res
            else:
                try:
                    res = await model.a_generate(prompt=prompt, schema=schema)
                    if res is None:
                        raise ValueError("Model returned None.")

                    if isinstance(res, str):
                        data = trimAndLoadJson(res)
                        return schema(**data)
                    else:
                        return res
                except TypeError:
                    res = await model.a_generate(prompt)
                    if res is None:
                        raise ValueError("Model returned None.")

                    data = trimAndLoadJson(res)
                    if schema == SyntheticDataList:
                        data_list = [SyntheticData(**item) for item in data["data"]]
                        return SyntheticDataList(data=data_list)
                    else:
                        return schema(**data)

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2**attempt
                logging.warning(
                    f"Async generation failed on attempt {attempt + 1}. Retrying in {sleep_time}s... Error: {e}"
                )
                await asyncio.sleep(sleep_time)

    raise RuntimeError(
        f"Failed to async generate after {MAX_RETRIES} attempts. Last error: {last_error}"
    )
