"""Cancellable HTTP transport shared by model and tool requests."""

import asyncio
import ipaddress
import socket

import httpx

from .budget import remaining_timeout
from .cancellation import OperationCancelled, check_cancelled, current_cancellation


class ResponseTooLargeError(ValueError):
    """A decoded HTTP response exceeded the caller's byte limit."""


class NonPublicAddressError(ValueError):
    """A restricted HTTP destination could not be resolved to public addresses."""


async def _public_address(url: httpx.URL) -> str:
    if url.scheme not in {"http", "https"} or not url.host:
        raise NonPublicAddressError("a public HTTP(S) destination is required")
    host = url.raw_host.decode("ascii")
    normalized_host = host.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise NonPublicAddressError("HTTP destination must be public")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            resolved = await asyncio.get_running_loop().getaddrinfo(
                host, url.port or (443 if url.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
            addresses = [ipaddress.ip_address(item[4][0]) for item in resolved]
        except (OSError, ValueError):
            raise NonPublicAddressError("HTTP destination could not be resolved") from None
    if not addresses or not all(address.is_global for address in addresses):
        raise NonPublicAddressError("HTTP destination must resolve only to public addresses")
    return str(addresses[0])


def send_request(
    request: httpx.Request, *, timeout: float,
    max_response_bytes: int | None, follow_redirects: bool = False,
    public_only: bool = False,
) -> httpx.Response:
    """Bound time and decoded body bytes; None explicitly leaves bytes unlimited."""
    if max_response_bytes is not None and (
        type(max_response_bytes) is not int or max_response_bytes < 1
    ):
        raise ValueError("max_response_bytes must be positive integer or None")
    check_cancelled()
    cancellation = current_cancellation()
    timeout = remaining_timeout(timeout)
    logical_urls: dict[httpx.Request, httpx.URL] = {}

    async def pin_public_address(pending: httpx.Request) -> None:
        logical_url = pending.url
        address = await _public_address(logical_url)
        logical_urls[pending] = logical_url
        pending.url = logical_url.copy_with(host=address)
        # The TCP destination is numeric; HTTP and TLS keep the logical host.
        pending.extensions["sni_hostname"] = logical_url.raw_host.decode("ascii")

    async def restore_logical_url(response: httpx.Response) -> None:
        pending = response.request
        pending.url = logical_urls.pop(pending)

    async def send() -> httpx.Response:
        async with asyncio.timeout(timeout):
            restricted_options = {}
            if public_only:
                restricted_options = {
                    "trust_env": False,
                    # Distinct TLS names can resolve to the same IP. Never reuse
                    # a connection authenticated for a previous logical host.
                    "limits": httpx.Limits(max_keepalive_connections=0),
                    "event_hooks": {
                        "request": [pin_public_address],
                        "response": [restore_logical_url],
                    },
                }
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=follow_redirects, max_redirects=5,
                **restricted_options,
            ) as client:
                if max_response_bytes is None:
                    response = await client.send(request)
                    response.raise_for_status()
                    return response
                pending = request
                history = []
                while True:
                    # httpx normally reads a redirect body before following it.
                    # Discard it here so redirects cannot bypass the byte limit.
                    response = await client.send(pending, stream=True, follow_redirects=False)
                    try:
                        if follow_redirects and response.next_request is not None:
                            if len(history) >= 5:
                                raise httpx.TooManyRedirects(
                                    "Exceeded 5 redirects", request=response.request,
                                )
                            history.append(response)
                            pending = response.next_request
                            continue
                        response.raise_for_status()
                        content = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            if len(content) + len(chunk) > max_response_bytes:
                                raise ResponseTooLargeError(
                                    f"HTTP response exceeds {max_response_bytes} decoded bytes"
                                )
                            content.extend(chunk)
                        # aiter_bytes already decoded Content-Encoding. Present
                        # the bounded body without asking httpx to decode twice.
                        headers = response.headers.copy()
                        headers.pop("content-encoding", None)
                        headers.pop("content-length", None)
                        return httpx.Response(
                            response.status_code, content=bytes(content), headers=headers,
                            request=response.request, extensions=response.extensions,
                            history=history,
                        )
                    finally:
                        await response.aclose()

    # asyncio.run waits for the default executor on shutdown. A cancelled DNS
    # lookup would therefore delay returning until getaddrinfo finishes. Closing
    # this per-request loop cancels the HTTP work without waiting for the resolver
    # thread; its eventual result cannot resume this request.
    loop = asyncio.new_event_loop()
    task = loop.create_task(send())

    def cancel_request() -> None:
        # Cancellation may arrive from the CLI/control reader's thread.
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            # The request already completed and its loop was closed.
            pass

    unregister = cancellation.on_cancel(cancel_request) if cancellation else lambda: None
    try:
        return loop.run_until_complete(task)
    except asyncio.CancelledError:
        if cancellation is not None and cancellation.cancelled:
            raise OperationCancelled(
                cancellation.reason,
                details={"operation": "http", "local_request_stopped": True,
                         "provider_execution_may_continue": True},
            ) from None
        raise
    finally:
        unregister()
        # Transport errors and cancellation do not reach the response hook.
        for pending, logical_url in logical_urls.items():
            pending.url = logical_url
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
