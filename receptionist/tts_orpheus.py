# receptionist/tts_orpheus.py
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace
from typing import Literal, Optional

import httpx

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import aio, is_given

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# Orpheus model specific settings
ORPHEUS_MODEL = "canopylabs/orpheus-v1-english"
ORPHEUS_API_URL = "https://api-inference.huggingface.co/models/" + ORPHEUS_MODEL


@dataclass
class _TTSOptions:
    model: str
    api_key: Optional[str]
    speed: float
    instructions: Optional[str]


class OrpheusTTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = ORPHEUS_MODEL,
        api_key: Optional[str] = None,
        speed: float = 1.0,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Orpheus TTS using Hugging Face Inference API.

        `api_key` must be set to your Hugging Face API key, either using the argument
        or by setting the `HF_API_KEY` environmental variable.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        # Determine API key from parameter or environment
        resolved_api_key = api_key
        if is_given(api_key):
            if not api_key:
                raise ValueError(
                    "Hugging Face API key is required, either as argument or set"
                    " HF_API_KEY environment variable"
                )
        else:
            resolved_api_key = os.environ.get("HF_API_KEY")
            if not resolved_api_key:
                raise ValueError(
                    "Hugging Face API key is required, either as argument or set"
                    " HF_API_KEY environment variable"
                )

        self._opts = _TTSOptions(
            model=model,
            api_key=resolved_api_key,
            speed=speed,
            instructions=instructions if is_given(instructions) else None,
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
            headers={
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def model(self) -> str:
        return self._opts.model

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[Optional[str]] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(api_key):
            if api_key is not None and not api_key:
                raise ValueError(
                    "Hugging Face API key is required, either as argument or set"
                    " HF_API_KEY environment variable"
                )
            self._opts.api_key = api_key
            # Update client headers if API key changed
            if api_key is not None:
                self._client.headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
        if is_given(speed):
            self._opts.speed = speed
        if is_given(instructions):
            self._opts.instructions = instructions

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._client.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: OrpheusTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: OrpheusTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Prepare payload for Hugging Face Inference API
        payload = {
            "inputs": self.input_text,
            "parameters": {
                "speed": self._opts.speed,
            }
        }
        if self._opts.instructions:
            payload["parameters"]["instructions"] = self._opts.instructions

        try:
            async with self._tts._client.stream(
                "POST",
                ORPHEUS_API_URL,
                json=payload,
                timeout=httpx.Timeout(30, connect=self._tts._client.timeout.connect),
            ) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    raise APIStatusError(
                        f"Hugging Face Inference API returned status {response.status_code}",
                        status_code=response.status_code,
                        body=error_content.decode(),
                    )

                # Get content type to determine how to process the response
                content_type = response.headers.get("content-type", "")

                # Initialize audio emitter
                output_emitter.initialize(
                    request_id="",
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/wav",  # Orpheus typically outputs WAV
                )

                # Process the audio data
                async for chunk in response.aiter_bytes():
                    output_emitter.push(chunk)

                output_emitter.flush()

        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except httpx.HTTPStatusError as e:
            raise APIStatusError(
                str(e),
                status_code=e.response.status_code,
                request_id=None,
                body=e.response.text,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e