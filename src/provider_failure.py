"""Shared Provider error type for the CLI and HTTP adapters."""


GEMINI_INTERACTIONS_HTTP_ERROR_CODES = frozenset({
    "invalid_request", "failed_precondition", "out_of_range", "parameter_unknown",
    "authentication", "payment_required", "permission_denied", "not_found",
    "model_not_found", "already_exists", "aborted", "rate_limit_exceeded",
    "quota_exceeded", "too_many_requests", "cancelled", "api_error",
    "unimplemented", "service_unavailable", "deadline_exceeded",
})


class ProviderFailure(Exception):
    def __init__(self, category, http_status=None, provider_error_code=None, measurements=None):
        super().__init__(category)
        self.category = category
        self.http_status = (http_status if type(http_status) is int and 100 <= http_status <= 599
                            else None)
        self.provider_error_code = (provider_error_code if isinstance(provider_error_code, str)
                                    and provider_error_code in GEMINI_INTERACTIONS_HTTP_ERROR_CODES
                                    else None)
        self.measurements = measurements
