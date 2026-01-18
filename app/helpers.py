from langchain_core.messages import BaseMessage
import json


def _extract_image_base64_from_response(response: object) -> str | None:
    if isinstance(response, dict) and "image_base64" in response:
        return response["image_base64"]

    if isinstance(response, list):
        for part in response:
            if isinstance(part, dict) and part.get("type") == "text":
                text_value = part.get("text", "")
                try:
                    payload_obj = json.loads(text_value)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload_obj, dict) and "image_base64" in payload_obj:
                    return payload_obj["image_base64"]

    if isinstance(response, str):
        try:
            payload_obj = json.loads(response)
        except json.JSONDecodeError:
            return None
        if isinstance(payload_obj, dict) and "image_base64" in payload_obj:
            return payload_obj["image_base64"]

    return None
