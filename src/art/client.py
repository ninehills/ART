import asyncio
import os
from typing import cast

import httpx
from openai import AsyncOpenAI, BaseModel, _exceptions
from openai._base_client import AsyncAPIClient
from openai._compat import cached_property
from openai._qs import Querystring
from openai._resource import AsyncAPIResource
from openai._types import Omit
from openai._utils import is_mapping
from openai._version import __version__
from openai.resources.models import AsyncModels  # noqa
from typing_extensions import override

from .trajectories import TrajectoryGroup


class Model(BaseModel):
    entity: str
    project: str
    name: str
    base_model: str

    @property
    def id(self) -> str:
        return f"{self.entity}/{self.project}/{self.name}"

    async def get_step(self) -> int:
        raise NotImplementedError("get_step is not implemented")
        # TODO: implement get_step
        return 0

    async def train(self, trajectory_groups: list[TrajectoryGroup]) -> None:
        raise NotImplementedError("train is not implemented")
        # TODO: implement train
        pass


class Models(AsyncAPIResource):
    async def create(
        self,
        *,
        entity: str | None = None,
        project: str | None = None,
        name: str | None = None,
        base_model: str,
    ) -> Model:
        model = await self._post(
            "/models",
            cast_to=Model,
            body={
                "entity": entity,
                "project": project,
                "name": name,
                "base_model": base_model,
            },
        )

        async def get_step() -> int:
            return 0

        model.get_step = get_step

        async def train(trajectory_groups: list[TrajectoryGroup]) -> None:
            training_job = await cast("Client", self._client).training_jobs.create(
                model_id=model.id,
                trajectory_groups=trajectory_groups,
            )
            while training_job.status != "COMPLETED":
                await asyncio.sleep(1)
                training_job = await cast(
                    "Client", self._client
                ).training_jobs.retrieve(training_job.id)

        model.train = train
        return model


class TrainingJob(BaseModel):
    id: int
    status: str


class TrainingJobs(AsyncAPIResource):
    async def create(
        self,
        *,
        model_id: str,
        trajectory_groups: list[TrajectoryGroup],
    ) -> TrainingJob:
        return await self._post(
            "/training-jobs",
            cast_to=TrainingJob,
            body={
                "model_id": model_id,
                "trajectory_groups": [
                    trajectory_group.model_dump()
                    for trajectory_group in trajectory_groups
                ],
            },
        )

    async def retrieve(self, training_job_id: int) -> TrainingJob:
        return await self._get(
            f"/training-jobs/{training_job_id}",
            cast_to=TrainingJob,
        )


class Client(AsyncAPIClient):
    api_key: str

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the WANDB_API_KEY environment variable"
            )
        self.api_key = api_key
        super().__init__(
            version=__version__,
            base_url=base_url or "http://0.0.0.0:8000/v1",
            _strict_response_validation=False,
        )

    @cached_property
    def models(self) -> Models:
        return Models(cast(AsyncOpenAI, self))

    @cached_property
    def training_jobs(self) -> TrainingJobs:
        return TrainingJobs(cast(AsyncOpenAI, self))

    ############################
    # AsyncOpenAI overrides #
    ############################

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="brackets")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        return {"Authorization": f"Bearer {api_key}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": "false",
            # "OpenAI-Organization": self.organization
            # if self.organization is not None
            # else Omit(),
            # "OpenAI-Project": self.project if self.project is not None else Omit(),
            **self._custom_headers,
        }

    @override
    def _make_status_error(
        self, err_msg: str, *, body: object, response: httpx.Response
    ) -> _exceptions.APIStatusError:
        data = body.get("error", body) if is_mapping(body) else body
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=data)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(
                err_msg, response=response, body=data
            )

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(
                err_msg, response=response, body=data
            )

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=data)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=data)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(
                err_msg, response=response, body=data
            )

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=data)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(
                err_msg, response=response, body=data
            )
        return _exceptions.APIStatusError(err_msg, response=response, body=data)
