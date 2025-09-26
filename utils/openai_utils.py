# openai_utils.py
from typing import Dict, Any
import os
import json
import re
import time

from openai import OpenAI
from types import SimpleNamespace
from google.api_core import exceptions 

# Vertex AI (Gemini on Google Cloud)
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel, GenerationConfig


def _wrap_chat_like(model: str, content: Any, usage: Any):
    um = usage or SimpleNamespace(prompt_token_count=0, candidates_token_count=0, total_token_count=0)
    return SimpleNamespace(
        id=None,
        object="chat.completion",
        model=model,
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(role="assistant", content=content),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
            total_tokens=int(getattr(um, "total_token_count", 0) or 0),
        ),
        provider="vertex-gemini",
    )

def _gemini_extract_text(vresp) -> str:
    # never call vresp.text
    if getattr(vresp, "candidates", None):
        c = vresp.candidates[0]
        parts = getattr(getattr(c, "content", None), "parts", None) or []
        buf = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                buf.append(t)
        if buf:
            return "".join(buf)
    return ""

_JSON_FIELD_RE = re.compile(r'"(?P<k>enriched_question|combined_enriched_question)"\s*:\s*"(?P<v>.*?)"', re.S)

def _strip_fences(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|text)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _maybe_extract_inner_enriched(s: str, fallback_key: str) -> str:
    """If s looks like a JSON blob, pull enriched text out of it."""
    if not isinstance(s, str): return ""
    s0 = _strip_fences(s)
    if s0.startswith("{") and s0.endswith("}"):
        # try JSON parse
        try:
            inner = json.loads(s0)
            if isinstance(inner, dict):
                for k in ("enriched_question", "combined_enriched_question", "question", "final"):
                    v = inner.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
        # regex fallback
        m = _JSON_FIELD_RE.search(s0)
        if m:
            return m.group("v").strip()
    return s.strip()


def create_response(
    stage: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    provider: str = "openai",  # "openai" or "gemini"
) -> Any:
    """
    Create a chat response using OpenAI or Gemini.

    Arguments:
        stage (str): pipeline stage selector
        prompt (str): user prompt
        model (str): model name (e.g., 'gpt-4o-mini' or 'gemini-1.5-pro')
        max_tokens (int): max generated tokens
        temperature (float): sampling temperature
        top_p (float): nucleus sampling
        n (int): number of completions (OpenAI supports n; Gemini returns 1)
        provider (str): 'openai' (default) or 'gemini'

    Returns:
        response_object (): OpenAI-like response object
                               with .choices[0].message.content
    """

    if stage == "question_enrichment":
        system_content = "You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions and the evidence to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items."
    elif stage == "candidate_sql_generation":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "sql_refinement":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "schema_filtering":
        system_content = "You are an excellent data scientist. You can capture the link between a question and corresponding database and determine the useful database items (tables and columns) perfectly. Your objective is to analyze and understand the essence of the given question, corresponding database schema, database column descriptions, samples and evidence and then select the useful database items such as tables and columns. This database item filtering is essential for eliminating unnecessary information in the database so that corresponding structured query language (SQL) of the question can be generated correctly in later steps."
    elif stage == "qe_combination":
        system_content = "You are an excellent data scientist who links information between a question and its database. You will read multiple enriched-question candidates along with the schema, descriptions, samples, evidence, and possible SQL conditions, and synthesize ONE clear, faithful combined enriched question. Do not invent tables/columns/values; keep names exactly as in the provided materials."
        
    else:
        raise ValueError("Wrong value for stage. It can only take following values: question_enrichment, candidate_sql_generation, sql_refinement, schema_filtering or qe_combination.")

    if provider.lower() == "openai":
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            temperature=temperature,
            top_p=top_p,
            n=n,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        # Return the original SDK object converted to a dict for consistency
        return resp

    elif provider.lower() == "gemini":
        
        project = os.getenv("GCP_PROJECT")
        env_loc = os.getenv("GCP_LOCATION")
        if not project:
            raise EnvironmentError("Set GCP_PROJECT and credentials in env")

        # init: prefer global for 2.5, then env, then common fallbacks
        tried = []
        for loc in (["global"] if model.startswith("gemini-2.5") else []) + ([env_loc] if env_loc else []) + ["us-east5","us-central1"]:
            if not loc: continue
            try:
                vertex_init(project=project, location=loc)
                break
            except Exception as e:
                tried.append((loc, str(e)))
        else:
            raise RuntimeError(f"Vertex init failed for all locations: {tried}")

        # strict JSON contract per stage so downstream always gets expected keys
        contracts = {
            "question_enrichment": 'Return ONLY JSON: {"chain_of_thought_reasoning": string, "enriched_question": string}',
            "candidate_sql_generation": 'Return ONLY JSON: {"chain_of_thought_reasoning": string, "SQL": string}',
            "sql_refinement": 'Return ONLY JSON: {"chain_of_thought_reasoning": string, "SQL": string}',
            "schema_filtering": 'Return ONLY JSON: {"useful_tables":[string], "useful_columns":[string]}',
            "qe_combination": 'Return ONLY JSON: {"combined_enrichment_reasoning": string, "combined_enriched_question": string}',
        }
        enforced_prompt = f"{prompt}\n\n{contracts.get(stage, '')}"

        m = GenerativeModel(model, system_instruction=system_content)
        cfg = GenerationConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            max_output_tokens=int(max_tokens) if max_tokens and max_tokens > 64 else 256,
            response_mime_type="application/json",
        )
        retries = 3  # Set the maximum number of retries
        delay = 10  # Set the initial delay in seconds (e.g., 10 seconds)
        
        while retries > 0:
            try:
                vresp = m.generate_content([enforced_prompt], generation_config=cfg)
                # If successful, break the loop
                break
            except exceptions.ResourceExhausted as e:
                print(f"ResourceExhausted error: {e}")
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                retries -= 1
            except Exception as e:
                # Handle other potential errors, or re-raise if not a ResourceExhausted error
                raise e
        else:
            # This block executes if the while loop finishes without a 'break'
            # meaning all retries were exhausted
            raise exceptions.ResourceExhausted("Failed to generate content after multiple retries due to resource exhaustion.")

        raw = _gemini_extract_text(vresp).strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
        if m: raw = m.group(1)

        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        # ── Stage-specific minimal normalization ──────────────────────────────────
        if stage in ("candidate_sql_generation", "sql_refinement"):
            # Ensure SQL key exists
            sql = payload.get("SQL")
            if not isinstance(sql, str) or not sql.strip():
                for k in ("sql_query","sql","predicted_sql","query"):
                    v = payload.get(k)
                    if isinstance(v, str) and v.strip():
                        sql = v; break
            if not sql:
                ms = re.search(r"```sql\s*(.*?)\s*```", raw, re.S|re.I) or re.search(r"(?is)\bSELECT\b[\s\S]+", raw)
                sql = (ms.group(1) if ms and ms.lastindex else (ms.group(0) if ms else "")).strip()
            payload["SQL"] = sql or ""

            # Always present for pipeline
            if not isinstance(payload.get("chain_of_thought_reasoning"), str):
                payload["chain_of_thought_reasoning"] = ""

        elif stage == "question_enrichment":
            # 1. Attempt to get enriched_question from the primary payload key
            eq = payload.get("enriched_question", "")
            
            # 2. Check if 'eq' looks like the full JSON object (malformed output)
            # If 'eq' starts with '{' or is excessively long (heuristic: > 512 chars), 
            # it likely contains the whole CoT, so we must re-extract the actual question.
            if isinstance(eq, str) and (eq.strip().startswith("{") or len(eq) > 512):
                # Use the helper function to try and extract the inner question from the malformed string
                extracted_eq = _maybe_extract_inner_enriched(eq, "enriched_question")
                if extracted_eq:
                    payload["enriched_question"] = extracted_eq
                else:
                    # Fallback if inner extraction fails on the malformed string
                    payload["enriched_question"] = (eq or "").strip()
            else:
                # If it didn't look malformed, just clean and set.
                payload["enriched_question"] = (eq or "").strip() 
                
            # 3. Final safety check: If enriched_question is still empty, 
            # try to extract it directly from the raw model output text.
            if not payload["enriched_question"]:
                 payload["enriched_question"] = _maybe_extract_inner_enriched(raw, "enriched_question")

            # 4. Ensure the reasoning field is present (even if empty string)
            if not isinstance(payload.get("chain_of_thought_reasoning"), str):
                payload["chain_of_thought_reasoning"] = ""

        elif stage == "qe_combination":
            ceq = payload.get("combined_enriched_question") or payload.get("enriched_question") or raw
            payload["combined_enriched_question"] = _maybe_extract_inner_enriched(ceq, "combined_enriched_question")
            cer = payload.get("combined_enrichment_reasoning")
            if not isinstance(cer, str):
                # try to salvage from raw if model emitted JSON-with-fences
                try:
                    raw_clean = _strip_fences(raw)
                    if raw_clean.startswith("{") and raw_clean.endswith("}"):
                        j = json.loads(raw_clean)
                        cer = j.get("combined_enrichment_reasoning", "")
                except Exception:
                    pass
            payload["combined_enrichment_reasoning"] = (cer or "").strip()


        elif stage == "schema_filtering":
            # Some Gemini responses use useful_{tables,columns}; map to template key
            if "tables_and_columns" not in payload or not isinstance(payload["tables_and_columns"], dict):
                ut = payload.get("useful_tables")
                uc = payload.get("useful_columns")
                if isinstance(ut, list) or isinstance(uc, list):
                    payload["tables_and_columns"] = {"__useful_tables__": ut or [], "__useful_columns__": uc or []}
                else:
                    payload["tables_and_columns"] = {}
            if not isinstance(payload.get("chain_of_thought_reasoning"), str):
                payload["chain_of_thought_reasoning"] = ""



        return _wrap_chat_like(model, payload, getattr(vresp, "usage_metadata", None))
        
    else:
        raise ValueError("Unsupported provider. Use 'openai' or 'gemini'.")


def upload_file_to_openai(file_path: str) -> Dict:
    """
    The function uploads given file to opanai for batch processing.

    Arguments:
        file_path (str): path of the file that is going to be uplaoded
    Returns:
        file_object (FileObject): Returned file object by openai
    """
    client = OpenAI()

    file_object = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )

    print("File is uploaded to OpenAI")
    return file_object


def construct_request_input_object(prompt: str, id: int, model: str, system_message: str) -> Dict:
    """
    The function creates a request input object for each item in the dataset

    Arguments:
        prompt (str): prompt that is going to given to the LLM as content
        id (int); the id of the request
        model (str): LLM model name
        system_message (str): the content of the system message

    Returns:
        request_input_object (Dict): The dictionary format required to be for request input
    """
    request_input_object = {
        "custom_id": f"qe-request-{id}",
        "method": "POST",
        "url": "/v1/chat/completions", 
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": f"{system_message}"}, 
                {"role": "user", "content": f"{prompt}"}
                ]
        }
    }
    return request_input_object