from __future__ import annotations


class ProviderError(RuntimeError):
    """Secret-safe failure raised by a provider implementation."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ProviderHTTPError(ProviderError):
    """An upstream provider rejected a request before a response could stream."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        code = _http_error_code(status_code)
        retryable = status_code in {408, 409, 425, 429} or status_code >= 500
        super().__init__(
            f"upstream provider returned HTTP {status_code}",
            code=code,
            retryable=retryable,
        )


class ProviderProtocolError(ProviderError):
    """An upstream response did not match the configured provider protocol."""

    def __init__(self, message: str = "upstream provider returned an invalid response") -> None:
        super().__init__(message, code="provider_protocol_error", retryable=False)


class ProviderTransportError(ProviderError):
    """The upstream provider could not be reached or stopped responding."""

    def __init__(self) -> None:
        super().__init__(
            "upstream provider transport failed",
            code="upstream_transport_error",
            retryable=True,
        )


def _http_error_code(status_code: int) -> str:
    if status_code in {401, 403}:
        return "upstream_authentication_error"
    if status_code in {408, 504}:
        return "upstream_timeout"
    if status_code == 429:
        return "upstream_rate_limited"
    if status_code >= 500:
        return "upstream_unavailable"
    return "upstream_rejected"
