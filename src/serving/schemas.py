from typing import Dict, Any

REQUIRED_FIELDS = [
    "age",
    "annual_income",
    "website_visits",
    "time_on_site",
    "discount_rate",
    "past_purchases",
    "region",
    "device_type",
    "membership_level"
]

def validate_request(payload: Dict[str, Any]):
    missing = set(REQUIRED_FIELDS) - set(payload.keys())
    if missing:
        raise ValueError(f"Missing fields: {missing}")


def validate_batch(payloads: Any):
    if not isinstance(payloads, list):
        raise ValueError("Batch payload must be a list of objects")
    if len(payloads) == 0:
        raise ValueError("Batch payload is empty")
    for idx, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ValueError(f"Item at index {idx} is not an object")
        validate_request(payload)
