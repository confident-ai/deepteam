from pydantic import BaseModel
from .schema import SyntheticData, SyntheticDataList

from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.models import DeepEvalBaseLLM


def generate(
    prompt: str,
    schema: BaseModel,
    model: DeepEvalBaseLLM = None,
) -> BaseModel:
    """
    Generate schema using the provided model.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        The validated schema object
    """
    _, using_native_model = initialize_model(model=model)
    if using_native_model:
        res, _ = model.generate(prompt=prompt, schema=schema)
        return res
    else:
        try:
            res = model.generate(prompt=prompt, schema=schema)
            if isinstance(res, str):
                data = trimAndLoadJson(res)
                return schema(**data)
            else:
                return res
        except TypeError:
            res = model.generate(prompt)
            data = trimAndLoadJson(res)
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)


async def a_generate(
    prompt: str,
    schema: BaseModel,
    model: DeepEvalBaseLLM = None,
) -> BaseModel:
    """
    Asynchronously generate schema using the provided model.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        The validated schema object
    """
    _, using_native_model = initialize_model(model=model)

    if using_native_model:
        res, _ = await model.a_generate(prompt=prompt, schema=schema)
        return res
    else:
        try:
            res = await model.a_generate(prompt=prompt, schema=schema)
            if isinstance(res, str):
                data = trimAndLoadJson(res)
                return schema(**data)
            else:
                return res
        except TypeError:
            res = await model.a_generate(prompt)
            data = trimAndLoadJson(res)
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)
