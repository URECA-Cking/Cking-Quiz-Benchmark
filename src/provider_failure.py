"""Shared Provider error type for the CLI and HTTP adapters."""


GEMINI_INTERACTIONS_HTTP_ERROR_CODES = frozenset({
    "invalid_request", "failed_precondition", "out_of_range", "parameter_unknown",
    "authentication", "payment_required", "permission_denied", "not_found",
    "model_not_found", "already_exists", "aborted", "rate_limit_exceeded",
    "quota_exceeded", "too_many_requests", "cancelled", "api_error",
    "unimplemented", "service_unavailable", "deadline_exceeded",
})


RETRY_STOP_REASONS = frozenset({"live_guard"})


class ProviderFailure(Exception):
    def __init__(self, category, http_status=None, provider_error_code=None, measurements=None,
                 retry_stop_reason=None):
        super().__init__(category)
        self.category = category
        self.http_status = (http_status if type(http_status) is int and 100 <= http_status <= 599
                            else None)
        self.provider_error_code = (provider_error_code if isinstance(provider_error_code, str)
                                    and provider_error_code in GEMINI_INTERACTIONS_HTTP_ERROR_CODES
                                    else None)
        self.measurements = measurements
        # Why a retry after an observed HTTP failure was not attempted; None otherwise.
        self.retry_stop_reason = (retry_stop_reason if isinstance(retry_stop_reason, str)
                                  and retry_stop_reason in RETRY_STOP_REASONS else None)
