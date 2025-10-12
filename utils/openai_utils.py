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

SQL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "chain_of_thought_reasoning": {
            "type": "STRING",
            "description": "The detailed, step-by-step logic."
        },
        "SQL": {
            "type": "STRING",
            "description": "The final, valid SQLite SQL query."
        },
    },
    "required": ["chain_of_thought_reasoning", "SQL"],
}

QE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "chain_of_thought_reasoning": {
            "type": "STRING",
            "description": "The step-by-step thinking process."
        },
        "enriched_question": {
            "type": "STRING",
            "description": "The final enriched question, rewritten with database details."
        },
    },
    "required": ["chain_of_thought_reasoning", "enriched_question"],
}

QE_SELECTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "selected_enrichment_reasoning": {
            "type": "string",
            "description": "The enrichment reasoning that produced the best enriched question."
        },
        "selected_enriched_question": {
            "type": "string",
            "description": "The single best enriched question to use, possibly with minor refinements."
        }
    },
    "required": ["selected_enrichment_reasoning", "selected_enriched_question"],
}

SCHEMA_FILTERING_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "chain_of_thought_reasoning": {
            "type": "STRING",
            "description": "The thinking process for item selection."
        },
        "useful_tables": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of relevant table names."
        },
        "useful_columns": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of relevant column names."
        },
    },
    "required": ["chain_of_thought_reasoning", "useful_tables", "useful_columns"],
}

ITERATIVE_ENRICH_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "enrichment_reasoning_v2": {
            "type": "STRING",
            "description": "Why v2 improves v1. Bullet style allowed."
        },
        "question_enriched_v2": {
            "type": "STRING",
            "description": "The refined one-step-better enriched question."
        }
    },
    "required": ["enrichment_reasoning_v2", "question_enriched_v2"],
}

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
    required_schema = None
    if stage == "question_enrichment":
        system_content = "You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions and the evidence to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items."
        required_schema = QE_SCHEMA
    elif stage == "candidate_sql_generation":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
        required_schema = SQL_SCHEMA
    elif stage == "sql_refinement":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
        required_schema = SQL_SCHEMA
    elif stage == "schema_filtering":
        system_content = "You are an excellent data scientist. You can capture the link between a question and corresponding database and determine the useful database items (tables and columns) perfectly. Your objective is to analyze and understand the essence of the given question, corresponding database schema, database column descriptions, samples and evidence and then select the useful database items such as tables and columns. This database item filtering is essential for eliminating unnecessary information in the database so that corresponding structured query language (SQL) of the question can be generated correctly in later steps."
        required_schema = SCHEMA_FILTERING_SCHEMA
    elif stage == "qe_combination":
        system_content = "You are an expert Data Scientist and a **Question Evaluation Specialist**. Your critical task is to rigorously evaluate all enriched question candidates by comparing them against the **Original Question's intent**, the database schema, and all provided evidence. Select the **single best candidate** that provides the most precise, faithful, and complete translation of the **Original Question** into database-centric terms. You may apply minor refinements to the selected question for accuracy, but you **must return the original reasoning** of the chosen candidate without any changes." # <--- FIX 3: Updated instructions
        required_schema = QE_SELECTION_SCHEMA
    elif stage == "iterative_enricher":
        system_content = "You are an excellent data scientist who takes an already enriched question (v1) and improves it one step further to produce v2. Validate every referenced table, column, and value against the schema and samples. Fix mismatches, resolve ambiguity, and specify joins, keys, filters, and aggregates only when supported. Use exact schema capitalization. Do not invent schema items. If v1 is already optimal, return it unchanged as v2 and justify briefly. Output JSON only."
        required_schema = ITERATIVE_ENRICH_SCHEMA
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
        
        if required_schema is None:
             raise ValueError(f"Schema object (e.g., QE_SCHEMA) is not defined for stage: {stage}")

        m = GenerativeModel(model, system_instruction=system_content)
        cfg = GenerationConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            max_output_tokens=int(max_tokens) if max_tokens and max_tokens > 64 else 256,
            response_mime_type="application/json",
            response_schema=required_schema,
        )
        retries = 3  # Set the maximum number of retries
        delay = 10  # Set the initial delay in seconds (e.g., 10 seconds)
        
        while retries > 0:
            try:
                vresp = m.generate_content([prompt], generation_config=cfg)
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

        try:
            payload = json.loads(raw) if raw and raw.startswith("{") else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        # ── Stage-specific minimal normalization ──────────────────────────────────
        if stage in ("candidate_sql_generation", "sql_refinement"):
            payload["SQL"] = payload.get("SQL", "").strip()
            payload["chain_of_thought_reasoning"] = payload.get("chain_of_thought_reasoning", "").strip()
            
        # For QE, only need minimal cleanup since schema is enforced
        elif stage == "question_enrichment":
            payload["enriched_question"] = payload.get("enriched_question", "").strip()
            payload["chain_of_thought_reasoning"] = payload.get("chain_of_thought_reasoning", "").strip()

        # For QE Combination, only need minimal cleanup
        elif stage == "qe_combination":
            # Note: The original helper _maybe_extract_inner_enriched is less necessary now
            payload["selected_enriched_question"] = payload.get("selected_enriched_question", "").strip()
            payload["selected_enrichment_reasoning"] = payload.get("selected_enrichment_reasoning", "").strip()
        
        elif stage == "iterative_enricher":
            payload["question_enriched_v2"] = payload.get("question_enriched_v2", "").strip()
            payload["enrichment_reasoning_v2"] = payload.get("enrichment_reasoning_v2", "").strip()

        # For Schema Filtering
        elif stage == "schema_filtering":
            # Map the clean output to the expected key for backward compatibility
            ut = payload.get("useful_tables")
            uc = payload.get("useful_columns")
            if isinstance(ut, list) or isinstance(uc, list):
                payload["tables_and_columns"] = {"__useful_tables__": ut or [], "__useful_columns__": uc or []}
            else:
                payload["tables_and_columns"] = {}
            payload["chain_of_thought_reasoning"] = payload.get("chain_of_thought_reasoning", "").strip()
            
        # Wrap and return the OpenAI-like object
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