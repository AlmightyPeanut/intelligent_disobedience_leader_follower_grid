from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


from dotenv import load_dotenv


# -----------------------
# Config & HTTP utilities
# -----------------------

@dataclass(frozen=True)
class ClientConfig:
    endpoint: str
    api_key: str
    timeout: float = 118.0  # seconds, applied to both connect & read


    @staticmethod
    def from_env() -> "ClientConfig":

        # Explicitly load .env from current working directory
        cwd_env = Path.cwd() / ".env"
        load_dotenv(dotenv_path=cwd_env, override=True)

        endpoint = os.getenv("LLMPROXY_ENDPOINT")
        api_key  = os.getenv("LLMPROXY_API_KEY")

        if not endpoint or not api_key:
            raise ValueError(
                "LLMProxy configuration error:\n"
                "Missing LLMPROXY_ENDPOINT or LLMPROXY_API_KEY.\n\n"
                "Make sure your .env file is in the SAME DIRECTORY where you run python.\n"
                "\nExample .env:\n"
                "    LLMPROXY_ENDPOINT=https://your-endpoint\n"
                "    LLMPROXY_API_KEY=your-api-key\n"
            )

        return ClientConfig(endpoint=endpoint, api_key=api_key)




def _build_session() -> requests.Session:
    """Session with retries and connection pooling."""
    s = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# -----------------------
# Core client
# -----------------------

class LLMProxy:
    def __init__(self) -> None:
        self.config = ClientConfig.from_env()
        self.session = _build_session()

    def _headers(self, request_type: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        base = {
            "x-api-key": self.config.api_key,
            "request_type": request_type,
        }
        if extra:
            base.update(extra)
        return base

    def _post_json(
        self,
        request_type: str,
        payload: Dict[str, Any],
    ) -> Dict:
        # Remove None values to avoid sending nulls unnecessarily
        clean_payload = {k: v for k, v in payload.items() if v is not None}

        try:
            resp = self.session.post(
                self.config.endpoint,
                headers=self._headers(request_type),
                json=clean_payload,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {e}", "status_code": None}

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                # JSON decode failed; return text for visibility
                return {"error": "Invalid JSON in response", "status_code": resp.status_code}
        else:
            # Try to surface server-provided error details
            detail: str
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            return {"error": f"HTTP {resp.status_code}: {detail}", "status_code": resp.status_code}

    # -------- Public methods --------

    def retrieve(
        self,
        query: str,
        session_id: str,
        rag_threshold: float,
        rag_k: int,
    ) -> Dict:
        """
        Calls the retrieval endpoint. Returns server JSON
        """
        payload = {
            "query": query,
            "session_id": session_id,
            "rag_threshold": rag_threshold,
            "rag_k": rag_k,
        }
        return self._post_json("retrieve", payload)

    def model_info(self) -> Dict:
        """
        Fetches model info.
        """
        return self._post_json("model_info", {})

    def generate(
        self,
        model: str,
        system: str,
        query: str,
        temperature: Optional[float] = None,
        lastk: Optional[int] = None,
        session_id: Optional[str] = "GenericSession",
        rag_threshold: Optional[float] = 0.5,
        rag_usage: Optional[bool] = False,
        rag_k: Optional[int] = 5,
    ) -> Dict:
        """
        Calls the text generation endpoint and returns parsed fields plus the raw payload.
        """
        payload = {
            "model": model,
            "system": system,
            "query": query,
            "temperature": temperature,
            "lastk": lastk,
            "session_id": session_id,
            "rag_threshold": rag_threshold,
            "rag_usage": rag_usage,
            "rag_k": rag_k,
        }
        res = self._post_json("call", payload)
        if "error" in res:
            return res
        # Defensive extraction
        return res
        # result_text = res.get("result")
        # rag_context = res.get("rag_context")
        # return {"response": result_text, "rag_context": rag_context, "raw": res}

    def upload_file(
        self,
        file_path: Union[str, Path],
        session_id: str,
        mime_type: str = None,
        description: Optional[str] = None,
        strategy: Optional[str] = "smart",
    ) -> Dict:
        """
        Generic uploader for any file. Uses streaming upload and returns server JSON or error.
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {path}", "status_code": None}

        if mime_type is None:
            # Minimal sniffing; caller can override
            mime_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"

        params = {
            "description": description,
            "session_id": session_id,
            "strategy": strategy,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        files = {
            # Include a filename so the server can store it meaningfully
            "params": (None, json.dumps(params), "application/json"),
            "file": (None, path.open("rb"), mime_type),
        }

        try:
            resp = self.session.post(
                self.config.endpoint,
                headers=self._headers("add"),
                files=files,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {e}", "status_code": None}

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                # If server returns plain text success
                return {"message": resp.text}
        else:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            return {"error": f"HTTP {resp.status_code}: {detail}", "status_code": resp.status_code}

    def upload_text(
        self,
        text: str,
        session_id: str,
        description: Optional[str] = None,
        strategy: Optional[str] = "smart",
    ) -> Dict:
        """
        Uploads raw text content as a 'file' part.
        """
        params = {
            "description": description,
            "session_id": session_id,
            "strategy": strategy,
        }
        params = {k: v for k, v in params.items() if v is not None}


        files = {
            "params": (None, json.dumps(params), "application/json"),            
            "text": (None, text, "application/text"),
        }

        try:
            resp = self.session.post(
                self.config.endpoint,
                headers=self._headers("add"),
                files=files,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {e}", "status_code": None}

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                return {"message": resp.text}
        else:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            return {"error": f"HTTP {resp.status_code}: {detail}", "status_code": resp.status_code}