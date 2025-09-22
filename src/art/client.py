from typing import cast

from openai._base_client import AsyncAPIClient
from openai._compat import cached_property
from openai._resource import AsyncAPIResource
from openai._version import __version__
from openai.resources.models import AsyncModels  # noqa

from openai import AsyncOpenAI


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
    api_key: str | None
    base_url: str | None

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(
            version=__version__,
            base_url=base_url or "http://0.0.0.0:8000",
            _strict_response_validation=False,
        )

    @cached_property
    def models(self) -> Models:
        return Models(cast(AsyncOpenAI, self))
