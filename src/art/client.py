import os
from typing import cast

import httpx
from openai._base_client import AsyncAPIClient
from openai._compat import cached_property
from openai._qs import Querystring
from openai._resource import AsyncAPIResource
from openai._types import Omit
from openai._utils import is_mapping
from openai._version import __version__
from openai.resources.models import AsyncModels  # noqa
from typing_extensions import override

from openai import AsyncOpenAI, _exceptions


class Model:
    def __init__(self, id: str) -> None:
        self.id = id


class Models(AsyncAPIResource):
    async def create(
        self,
        *,
        entity: str | None = None,
        project: str | None = None,
        name: str | None = None,
        base_model: str,
    ) -> Model:
        return await self._post(
            "/models",
            cast_to=Model,
            body={
                "entity": entity,
                "project": project,
                "name": name,
                "base_model": base_model,
            },
        )


class Client(AsyncAPIClient):
    api_key: str
    base_url: str

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
        self.base_url = base_url or "http://0.0.0.0:8000/v1"
        super().__init__(
            version=__version__,
            base_url=self.base_url,
            _strict_response_validation=False,
        )

    @cached_property
    def models(self) -> Models:
        return Models(cast(AsyncOpenAI, self))

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
