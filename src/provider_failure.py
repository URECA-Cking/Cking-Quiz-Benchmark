"""Shared Provider error type for the CLI and HTTP adapters."""


class ProviderFailure(Exception):
    def __init__(self, category, http_status=None):
        super().__init__(category)
        self.category = category
        self.http_status = (http_status if type(http_status) is int and 100 <= http_status <= 599
                            else None)
