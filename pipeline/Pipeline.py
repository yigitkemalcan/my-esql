import os
import json
from utils.prompt_utils import *
from utils.db_utils import * 
from utils.openai_utils import create_response
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.retrieval_utils import query_vector_db, query_db_values
import sqlite3

class Pipeline():
    def __init__(self, args):
        # Running mode attributes
        self.mode = args.mode 
        self.dataset_path = args.dataset_path

        # Pipeline attribute
        self.pipeline_order = args.pipeline_order

        # Model attributes
        self.model = args.model
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_tokens = args.max_tokens
        self.n = args.n

        # Stages (enrichment, filtering, generation) attributes
        self.enrichment_level = args.enrichment_level
        self.elsn = args.enrichment_level_shot_number
        self.efsse = args.enrichment_few_shot_schema_existance

        self.flsn = args.filtering_level_shot_number
        self.ffsse = args.filtering_few_shot_schema_existance

        self.cfg = args.cfg
        self.glsn = args.generation_level_shot_number
        self.gfsse = args.generation_few_shot_schema_existance

        self.db_sample_limit = args.db_sample_limit
        self.rdn = args.relevant_description_number

        self.seed = args.seed

        self.num_enriched_questions = getattr(args, "num_enriched_questions", 1)
        self.provider = getattr(args, "provider", "openai")
        self.iterative_enricher = getattr(args, "iterative_enricher", False)

    def convert_message_content_to_dict(self, response_object: Dict) -> Dict:
        """
        The function gets a LLM response object, and then it converts the content of it to the Python object.

        Arguments:
            response_object (Dict): LLM response object
        Returns:
            response_object (Dict): Response object whose content changed to dictionary
        """

        response_object.choices[0].message.content = json.loads(response_object.choices[0].message.content)
        return response_object
    

    def forward_pipeline_CSG_SR(self, t2s_object: Dict) -> Dict[str, str]:
        """
        The function runs Candidate SQL Generation(CSG) and SQL Refinement(SR) modules respectively without any question enrichment or filtering stages.

        Arguments:
            t2s_object (Dict): Python dictionary that stores information about a question like q_id, db_id, question, evidence etc. 
        Returns:
            t2s_object_prediction (Dict):  Python dictionary that stores information about a question like q_id, db_id, question, evidence etc and also stores information after each stage
        """
        db_id = t2s_object["db_id"]
        q_id = t2s_object["question_id"]
        evidence = t2s_object["evidence"]
        question = t2s_object["question"]
        
        bird_sql_path = os.getenv('BIRD_DB_PATH')
        db_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/{db_id}.sqlite"
        db_description_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/database_description"
        db_descriptions = question_relevant_descriptions_prep(database_description_path=db_description_path, question=question, relevant_description_number=self.rdn)
        database_column_meaning_path = bird_sql_path + f"/{self.mode}/column_meaning.json"
        db_column_meanings = db_column_meaning_prep(database_column_meaning_path, db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # extracting original schema dictionary 
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=db_path)

        t2s_object["question_enrichment"] = "No Question Enrichment"

        ### STAGE 1: Candidate SQL GENERATION
        # -- Original question is used
        # -- Original Schema is used 
        sql_generation_response_obj =  self.candidate_sql_generation_module(db_path=db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=original_schema_dict, db_descriptions=db_descriptions)
        try:
            possible_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "possible_sql": possible_sql,
                "exec_err": "",
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["possible_sql"] = possible_sql
            # execute SQL
            try:
                possible_respose = func_timeout(30, execute_sql, args=(db_path, possible_sql))
            except FunctionTimedOut:
                t2s_object['candidate_sql_generation']["exec_err"] = "timeout"
            except Exception as e:
                t2s_object['candidate_sql_generation']["exec_err"] = str(e)
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": "",
                "possible_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["candidate_sql_generation"]["error"] = f"{e}"
            return t2s_object
        
        ### STAGE 2: SQL Refinement
        # -- Original question is used
        # -- Original Schema is used 
        # -- Possible SQL is used
        # -- Possible Conditions is extracted from possible SQL and then used for augmentation
        # -- Execution Error for Possible SQL is used
        exec_err = t2s_object['candidate_sql_generation']["exec_err"]
        sql_generation_response_obj =  self.sql_refinement_module(db_path=db_path, db_id=db_id, question=question, evidence=evidence, possible_sql=possible_sql, exec_err=exec_err, filtered_schema_dict=original_schema_dict, db_descriptions=db_descriptions)
        try:
            predicted_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "predicted_sql": predicted_sql,
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["predicted_sql"] = predicted_sql
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": "",
                "predicted_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["sql_refinement"]["error"] = f"{e}"
            return t2s_object

        # storing the usage for one question
        t2s_object["total_usage"] = {
            "prompt_tokens": t2s_object['candidate_sql_generation']['prompt_tokens'] + t2s_object['sql_refinement']['prompt_tokens'],
            "completion_tokens": t2s_object['candidate_sql_generation']['completion_tokens'] + t2s_object['sql_refinement']['completion_tokens'],
            "total_tokens": t2s_object['candidate_sql_generation']['total_tokens'] + t2s_object['sql_refinement']['total_tokens']
        }

        t2s_object_prediction = t2s_object
        return t2s_object_prediction
    
    def forward_pipeline_CSG_QE_SR(self, t2s_object: Dict) -> Dict:
        """
        The function performs Candidate SQL Generation(CSG), Quesiton Enrichment(QE) and SQL Refinement(SR) modules respectively without filtering stages.
        
        Arguments:
            t2s_object (Dict): Python dictionary that stores information about a question like q_id, db_id, question, evidence etc. 
        Returns:
            t2s_object_prediction (Dict):  Python dictionary that stores information about a question like q_id, db_id, question, evidence etc and also stores information after each stage
        """
        db_id = t2s_object["db_id"]
        q_id = t2s_object["question_id"]
        evidence = t2s_object["evidence"]
        question = t2s_object["question"]
        
        bird_sql_path = os.getenv('BIRD_DB_PATH')
        db_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/{db_id}.sqlite"
        db_description_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/database_description"
        db_descriptions = question_relevant_descriptions_prep(database_description_path=db_description_path, question=question, relevant_description_number=self.rdn)
        database_column_meaning_path = bird_sql_path + f"/{self.mode}/column_meaning.json"
        db_column_meanings = db_column_meaning_prep(database_column_meaning_path, db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # extracting original schema dictionary 
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=db_path)

        ### STAGE 1: Candidate SQL GENERATION
        # -- Original question is used
        # -- Original Schema is used 
        sql_generation_response_obj =  self.candidate_sql_generation_module(db_path=db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=original_schema_dict, db_descriptions=db_descriptions)
        try:
            possible_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "possible_sql": possible_sql,
                "exec_err": "",
                "prompt_tokens": int((sql_generation_response_obj.usage.prompt_tokens or 0)),
                "completion_tokens": int((sql_generation_response_obj.usage.completion_tokens or 0)),
                "total_tokens": int((sql_generation_response_obj.usage.total_tokens or 0)),
            }
            t2s_object["possible_sql"] = possible_sql
            # execute SQL
            try:
                possible_respose = func_timeout(30, execute_sql, args=(db_path, possible_sql))
            except FunctionTimedOut:
                t2s_object['candidate_sql_generation']["exec_err"] = "timeout"
            except Exception as e:
                t2s_object['candidate_sql_generation']["exec_err"] = str(e)
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": "",
                "possible_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["candidate_sql_generation"]["error"] = f"{e}"
            return t2s_object
        
        # Extract possible conditions dict list
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=possible_sql)
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)

        ### STAGE 2: Question Enrichment (MULTI + COMBINE)
        # -- Original question is used
        # -- Original schema is used
        # -- Possible conditions are used
        # Run QE N times concurrently
        num_eq = getattr(self, "num_enriched_questions", 1)
        enriched_variants = []
        qe_runs_usage = []
        qe_first_reasoning = ""
        qe_first_enriched = ""

        def _qe_call(i: int):
            resp = self.question_enrichment_module(
                db_path=db_path, q_id=q_id, db_id=db_id,
                question=question, evidence=evidence,
                possible_conditions=possible_conditions,
                schema_dict=original_schema_dict,
                db_descriptions=db_descriptions
            )
            return i, resp

        workers = 1
        results = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_qe_call, i) for i in range(num_eq)]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception:
                    # ignore failed attempt; fallbacks below
                    pass

        # Preserve deterministic "first run" semantics by index
        for i, q_enrich_response_obj in sorted(results, key=lambda x: x[0]):
            try:
                eq = q_enrich_response_obj.choices[0].message.content['enriched_question']
                er = q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning']
                if isinstance(eq, str) and eq.strip():
                    enriched_variants.append({
                        "enriched_question": eq.strip(),
                        "enrichment_reasoning": er.strip()
                    })

                um = getattr(q_enrich_response_obj, "usage", None)
                qe_runs_usage.append({
                    "prompt_tokens": int(getattr(um, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(um, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(um, "total_tokens", 0) or 0),
                })

                if qe_first_enriched == "":
                    qe_first_enriched = eq
                    qe_first_reasoning = er
            except Exception as e:
                logging.error(f"Error parsing QE response for qid {q_id} run {i}: {e}")
                continue


        # For observability: keep all generated variants
        t2s_object["question_enrichment_variants"] = enriched_variants

        # Fallback if nothing returned
        if not enriched_variants:
            # preserve original structure with empties
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": "",
                "enriched_question": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            # final enriched for SR falls back to original question
            enriched_question_final = question

        # If only one variant, keep the original Experiment-24 concatenation behavior
        elif len(enriched_variants) == 1:
            t2s_object["question_enrichment_first_run"] = {
                "enrichment_reasoning": qe_first_reasoning,
                "enriched_question": qe_first_enriched,
                "prompt_tokens": qe_runs_usage[0]["prompt_tokens"] if qe_runs_usage else 0,
                "completion_tokens": qe_runs_usage[0]["completion_tokens"] if qe_runs_usage else 0,
                "total_tokens": qe_runs_usage[0]["total_tokens"] if qe_runs_usage else 0,
            }
            if self.iterative_enricher and self.num_enriched_questions == 1:
                iter_resp = self.iterative_enricher_module(
                    db_path=db_path,
                    q_id=q_id,
                    db_id=db_id,
                    question=question,
                    enriched_question_v1=qe_first_enriched,
                    enrichment_reasoning_v1=qe_first_reasoning,
                    evidence=evidence,
                    possible_conditions=possible_conditions,
                    schema_dict=original_schema_dict,
                    db_descriptions=db_descriptions
                )
                try:
                    content = iter_resp.choices[0].message.content or {}
                    v2_text = content.get("question_enriched_v2", "").strip()
                    v2_reason = content.get("enrichment_reasoning_v2", "").strip()

                    t2s_object["iterative_enricher"] = {
                        "question_enriched_v2": v2_text,
                        "enrichment_reasoning_v2": v2_reason,
                        "prompt_tokens": int(getattr(iter_resp.usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(iter_resp.usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(iter_resp.usage, "total_tokens", 0) or 0),
                    }

                    if v2_text:
                        enriched_question_final = question + "\n" + v2_reason + "\n" + v2_text
                    else:
                        enriched_question_final = question + "\n" + qe_first_reasoning + "\n" + qe_first_enriched
                except Exception:
                    enriched_question_final = question + "\n" + qe_first_reasoning + "\n" + qe_first_enriched

            else:
                enriched_question_final = question + "\n" + qe_first_reasoning + "\n" + qe_first_enriched

        # If multiple variants, call the new COMBINATION module
        else:
            comb_response_obj = self.qe_combination_module(
                db_path=db_path, q_id=q_id, db_id=db_id,
                question=question, evidence=evidence,
                possible_conditions=possible_conditions,
                schema_dict=original_schema_dict,
                db_descriptions=db_descriptions,
                enriched_questions_with_reasoning=enriched_variants
            )
            try:
                content = comb_response_obj.choices[0].message.content or {}
                combined_enrichment_reasoning = content.get('selected_enrichment_reasoning', '')
                combined_enriched_question = content.get('selected_enriched_question', '')
                # Record the combiner call usage
                t2s_object["qe_selection"] = {
                    "selected_enrichment_reasoning": combined_enrichment_reasoning,
                    "selected_enriched_question": combined_enriched_question,
                    "prompt_tokens": int((comb_response_obj.usage.prompt_tokens or 0)),
                    "completion_tokens": int((comb_response_obj.usage.completion_tokens or 0)),
                    "total_tokens": int((comb_response_obj.usage.total_tokens or 0)),
                }

                # For backward compatibility: keep first QE run as the canonical QE record
                # (tools downstream may expect these fields to exist)
                if qe_runs_usage:
                    t2s_object["question_enrichment"] = {
                        "enrichment_reasoning": qe_first_reasoning,
                        "enriched_question": qe_first_enriched,
                        "prompt_tokens": int(qe_runs_usage[0]["prompt_tokens"]) if qe_runs_usage else 0,
                        "completion_tokens": int(qe_runs_usage[0]["completion_tokens"]) if qe_runs_usage else 0,
                        "total_tokens": int(qe_runs_usage[0]["total_tokens"]) if qe_runs_usage else 0,
                    }

                # There is no reasoning in the combiner output; to stay aligned with Experiment-24
                # we prepend the original question. (You can drop this if you want the pure combined text.)
                enriched_question_final = (
                    question + "\n" +
                    (combined_enrichment_reasoning.strip() + "\n" if combined_enrichment_reasoning else "") +
                    combined_enriched_question
                )

            except Exception as e:
                logging.error(f"Error in reaching content from QE combination response for question_id {q_id}: {e}")
                # Fallback: use first QE run’s concatenation
                t2s_object["qe_combination"] = {
                    "combined_enriched_question": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "error": f"{e}",
                }
                enriched_question_final = question + "\n" + qe_first_reasoning + "\n" + qe_first_enriched

        
        ### STAGE 3: SQL Refinement
        # -- Enriched question is used
        # -- Original Schema is used 
        # -- Possible SQL is used
        # -- Possible Conditions is extracted from possible SQL and then used for augmentation
        # -- Execution Error for Possible SQL is used
        exec_err = t2s_object['candidate_sql_generation']["exec_err"]
        sql_generation_response_obj =  self.sql_refinement_module(db_path=db_path, db_id=db_id, question=enriched_question_final, evidence=evidence, possible_sql=possible_sql, exec_err=exec_err, filtered_schema_dict=original_schema_dict, db_descriptions=db_descriptions)
        try:
            predicted_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "predicted_sql": predicted_sql,
                "prompt_tokens": int((sql_generation_response_obj.usage.prompt_tokens or 0)),
                "completion_tokens": int((sql_generation_response_obj.usage.completion_tokens or 0)),
                "total_tokens": int((sql_generation_response_obj.usage.total_tokens or 0)),
            }
            t2s_object["predicted_sql"] = predicted_sql
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": "",
                "predicted_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["sql_refinement"]["error"] = f"{e}"
            return t2s_object

        # storing the usage for one question
        # Sum all QE calls + optional combiner; keep backward compat if lists missing
        qe_prompt_sum = sum((u.get("prompt_tokens") or 0) for u in qe_runs_usage) if 'qe_runs_usage' in locals() else int(t2s_object['question_enrichment'].get('prompt_tokens', 0) or 0)
        qe_completion_sum = sum((u.get("completion_tokens") or 0) for u in qe_runs_usage) if 'qe_runs_usage' in locals() else int(t2s_object['question_enrichment'].get('completion_tokens', 0) or 0)
        qe_total_sum = sum((u.get("total_tokens") or 0) for u in qe_runs_usage) if 'qe_runs_usage' in locals() else int(t2s_object['question_enrichment'].get('total_tokens', 0) or 0)


        comb_prompt = int(t2s_object.get("qe_combination", {}).get("prompt_tokens", 0) or 0)
        comb_completion = int(t2s_object.get("qe_combination", {}).get("completion_tokens", 0) or 0)
        comb_total = int(t2s_object.get("qe_combination", {}).get("total_tokens", 0) or 0)


        t2s_object["total_usage"] = {
            "prompt_tokens": t2s_object['candidate_sql_generation']['prompt_tokens'] + qe_prompt_sum + comb_prompt + t2s_object['sql_refinement']['prompt_tokens'],
            "completion_tokens": t2s_object['candidate_sql_generation']['completion_tokens'] + qe_completion_sum + comb_completion + t2s_object['sql_refinement']['completion_tokens'],
            "total_tokens": t2s_object['candidate_sql_generation']['total_tokens'] + qe_total_sum + comb_total + t2s_object['sql_refinement']['total_tokens']
        }

        t2s_object_prediction = t2s_object
        return t2s_object_prediction
    

    def forward_pipeline_SF_CSG_QE_SR(self, t2s_object: Dict) -> Dict:
        """
        The function performs, Schema Filtering(SF) Candidate SQL Generation(CSG), Quesiton Enrichment(QE) and SQL Refinement(SR) modules respectively without filtering stages.
        
        Arguments:
            t2s_object (Dict): Python dictionary that stores information about a question like q_id, db_id, question, evidence etc. 
        Returns:
            t2s_object_prediction (Dict):  Python dictionary that stores information about a question like q_id, db_id, question, evidence etc and also stores information after each stage
        """
        db_id = t2s_object["db_id"]
        q_id = t2s_object["question_id"]
        question = t2s_object["question"]
        evidence = t2s_object.get("evidence", "")
        # Prepare database paths
        bird_sql_path = os.getenv('BIRD_DB_PATH', self.dataset_path)
        db_dir = f"{bird_sql_path}/{self.mode}/{self.mode}_databases/{db_id}"
        db_path = f"{db_dir}/{db_id}.sqlite"
        db_description_path = f"{db_dir}/database_description"
        # Stage 0: Information Retrieval
        keywords_result = self.extract_keywords_module(question=question, hint=evidence)
        try:
            kw_content = keywords_result.choices[0].message.content or {}
            keywords = kw_content.get("keywords", [])
        except Exception:
            keywords = []
        t2s_object["keywords"] = keywords
        similar_entities = {}
        try:
            similar_entities = query_db_values(
                db_directory_path=db_dir,
                keywords=keywords
            )
        except Exception as e:
            logging.error(f"Value retrieval error: {e}")

        t2s_object["similar_entities"] = similar_entities

        # Construct HINT from retrieved values (flatten the dict structure)
        hint_lines = []
        for table_name, col_dict in similar_entities.items():
            for col_name, values in col_dict.items():
                for value in values:
                    if value is None:
                        continue
                    val_str = str(value)
                    if val_str.lower() in question.lower():
                        idx = question.lower().find(val_str.lower())
                        phrase = question[max(0, idx - 20): idx + len(val_str) + 5]
                        hint_lines.append(
                            f"{phrase.strip()} refers to {table_name}.{col_name} = {val_str};"
                        )
                    else:
                        hint_lines.append(
                            f"'{val_str}' is a value of {table_name}.{col_name};"
                        )
        hint_text = " ".join(hint_lines)
        combined_hint = (evidence.strip() + "\n" + hint_text) if evidence and hint_text else (evidence or hint_text or "No hint")
        t2s_object["evidence"] = combined_hint
        evidence = combined_hint
        # Combine relevant descriptions and column meanings for generation stage
        db_descriptions = relevant_descriptions_prep(
            database_description_path=db_description_path,
            question=question,
            relevant_description_number=self.rdn
        )
        try:
            col_meaning_file = f"{bird_sql_path}/{self.mode}/column_meaning.json"
            with open(col_meaning_file, 'r') as f:
                col_meanings_all = json.load(f)
            if db_id in col_meanings_all:
                meanings_text = "\n".join([f"{col}: {meaning}" for col, meaning in col_meanings_all[db_id].items()])
                db_descriptions = (db_descriptions + "\n\n" + meanings_text) if db_descriptions else meanings_text
        except Exception:
            pass
        # Stage 1: Schema Filtering (IR + SS)
        filtered_schema = self.filter_columns_module(question=question, hint=combined_hint, db_path=db_path)
        t2s_object["filtered_columns"] = filtered_schema
        selected_tables = self.select_tables_module(question=question, hint=combined_hint, filtered_schema=filtered_schema)
        t2s_object["selected_tables"] = selected_tables
        selected_schema = self.select_columns_module(question=question, hint=combined_hint,
                                                     selected_tables=selected_tables, filtered_schema=filtered_schema)
        t2s_object["selected_columns"] = selected_schema

        ### STAGE 2: Candidate SQL GENERATION
        # -- Original question is used
        # -- Filtered Schema is used 
        sql_generation_response_obj =  self.candidate_sql_generation_module(db_path=db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=selected_schema, db_descriptions=db_descriptions)
        try:
            possible_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "possible_sql": possible_sql,
                "exec_err": "",
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["possible_sql"] = possible_sql
            # execute SQL
            try:
                possible_respose = func_timeout(30, execute_sql, args=(db_path, possible_sql))
            except FunctionTimedOut:
                t2s_object['candidate_sql_generation']["exec_err"] = "timeout"
            except Exception as e:
                t2s_object['candidate_sql_generation']["exec_err"] = str(e)
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": "",
                "possible_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["candidate_sql_generation"]["error"] = f"{e}"
            return t2s_object
        
        # Extract possible conditions dict list
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=possible_sql)
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)

        ### STAGE 3: Question Enrichment:
        # -- Original question is used
        # -- Original schema is used
        # -- Possible conditions are used
        q_enrich_response_obj = self.question_enrichment_module(db_path=db_path, q_id=q_id, db_id=db_id, question=question, evidence=evidence, possible_conditions=possible_conditions, schema_dict=selected_schema, db_descriptions=db_descriptions)
        try:
            enriched_question = q_enrich_response_obj.choices[0].message.content['enriched_question']
            enrichment_reasoning = q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning']
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "enriched_question": q_enrich_response_obj.choices[0].message.content['enriched_question'],
                "prompt_tokens": q_enrich_response_obj.usage.prompt_tokens,
                "completion_tokens": q_enrich_response_obj.usage.completion_tokens,
                "total_tokens": q_enrich_response_obj.usage.total_tokens,
            }
            enriched_question = question + enrichment_reasoning + enriched_question # This is added after experiment-24
        except Exception as e:
            logging.error(f"Error in reaching content from question enrichment response for question_id {q_id}: {e}")
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": "",
                "enriched_question": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["question_enrichment"]["error"] = f"{e}"
            enriched_question = question
        
        ### STAGE 4: SQL Refinement
        # -- Enriched question is used
        # -- Original Schema is used 
        # -- Possible SQL is used
        # -- Possible Conditions is extracted from possible SQL and then used for augmentation
        # -- Execution Error for Possible SQL is used
        exec_err = t2s_object['candidate_sql_generation']["exec_err"]
        sql_generation_response_obj =  self.sql_refinement_module(db_path=db_path, db_id=db_id, question=enriched_question, evidence=evidence, possible_sql=possible_sql, exec_err=exec_err, filtered_schema_dict=selected_schema, db_descriptions=db_descriptions)
        try:
            predicted_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "predicted_sql": predicted_sql,
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["predicted_sql"] = predicted_sql
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": "",
                "predicted_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["sql_refinement"]["error"] = f"{e}"
            return t2s_object

        # storing the usage for one question
        t2s_object["total_usage"] = {
            "prompt_tokens": t2s_object['candidate_sql_generation']['prompt_tokens'] + t2s_object['question_enrichment']['prompt_tokens'] + t2s_object['sql_refinement']['prompt_tokens'],
            "completion_tokens": t2s_object['candidate_sql_generation']['completion_tokens'] + t2s_object['question_enrichment']['completion_tokens'] + t2s_object['sql_refinement']['completion_tokens'],
            "total_tokens": t2s_object['candidate_sql_generation']['total_tokens'] + t2s_object['question_enrichment']['total_tokens'] + t2s_object['sql_refinement']['total_tokens']
        }

        t2s_object_prediction = t2s_object
        return t2s_object_prediction

        
    def construct_question_enrichment_prompt(self, db_path: str, q_id: int, db_id: str, question: str, evidence: str, possible_conditions: str, schema_dict: Dict, db_descriptions: str) -> str:
        """
        The function constructs the prompt required for the question enrichment stage

        Arguments:
            db_path (str): path to database sqlite file
            q_id (int): question id
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            possible_conditions (str): Possible conditions extracted from the previously generated possible SQL for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            prompt (str): Question enrichment prompt
        """
        enrichment_template_path = os.path.join(os.getcwd(), "prompt_templates/question_enrichment_prompt_template.txt")
        question_enrichment_prompt_template = extract_question_enrichment_prompt_template(enrichment_template_path)
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        q_enrich_few_shot_examples = question_enrichment_few_shot_prep(few_shot_data_path, q_id=q_id, q_db_id=db_id, level_shot_number=self.elsn, schema_existance=self.efsse, enrichment_level=self.enrichment_level, mode=self.mode)
        db_samples = extract_db_samples_enriched_bm25(question, evidence, db_path=db_path, schema_dict=schema_dict, sample_limit=self.db_sample_limit)
        schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)
        prompt = fill_question_enrichment_prompt_template(template=question_enrichment_prompt_template, schema=schema, db_samples=db_samples, question=question, possible_conditions=possible_conditions, few_shot_examples=q_enrich_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions)
        # print("question_enrichment_prompt: \n", prompt)
        return prompt
    
    def question_enrichment_module(self, db_path: str, q_id: int, db_id: str, question: str, evidence: str, possible_conditions: str, schema_dict: Dict, db_descriptions: str) -> Dict:
        """
        The function enrich the given question using LLM.

        Arguments:
            db_path (str): path to database sqlite file
            q_id (int): question id
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            possible_conditions (str): possible conditions extracted from previously generated possible SQL for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_question_enrichment_prompt(db_path=db_path, q_id=q_id, db_id=db_id, question=question, evidence=evidence, possible_conditions=possible_conditions, schema_dict=schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="question_enrichment", prompt=prompt, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=self.n, provider=self.provider)
        try:
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            return response_object

        return response_object
    
    def construct_candidate_sql_generation_prompt(self, db_path: str, db_id: int, question: str, evidence: str, filtered_schema_dict: Dict, db_descriptions: str)->str:
        """
        The function constructs the prompt required for the candidate SQL generation stage.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            prompt (str): prompt for SQL generation stage
        """
        sql_generation_template_path =  os.path.join(os.getcwd(), "prompt_templates/candidate_sql_generation_prompt_template.txt")
        with open(sql_generation_template_path, 'r') as f:
            sql_generation_template = f.read()
            
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        sql_generation_few_shot_examples = sql_generation_and_refinement_few_shot_prep(few_shot_data_path, q_db_id=db_id, level_shot_number=self.glsn, schema_existance=self.gfsse, mode=self.mode)
        db_samples = extract_db_samples_enriched_bm25(question, evidence, db_path, schema_dict=filtered_schema_dict, sample_limit=self.db_sample_limit)
        filtered_schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        prompt = fill_candidate_sql_prompt_template(template=sql_generation_template, schema=filtered_schema, db_samples=db_samples, question=question, few_shot_examples=sql_generation_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions) 
        # print("candidate_sql_prompt: \n", prompt)
        return prompt

    
    def construct_sql_refinement_prompt(self, db_path: str, db_id: int, question: str, evidence: str, possible_sql: str, exec_err: str, filtered_schema_dict: Dict, db_descriptions: str)->str:
        """
        The function constructs the prompt required for the SQL refinement stage.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            possible_sql (str): Previously generated possible SQL for the question
            exec_err (str): Taken execution error when possible SQL is executed
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            prompt (str): prompt for SQL generation stage
        """
        sql_generation_template_path =  os.path.join(os.getcwd(), "prompt_templates/sql_refinement_prompt_template.txt")
        with open(sql_generation_template_path, 'r') as f:
            sql_generation_template = f.read()
            
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        sql_generation_few_shot_examples = sql_generation_and_refinement_few_shot_prep(few_shot_data_path, q_db_id=db_id, level_shot_number=self.glsn, schema_existance=self.gfsse, mode=self.mode)
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=possible_sql)
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)
        filtered_schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        prompt = fill_refinement_prompt_template(template=sql_generation_template, schema=filtered_schema, possible_conditions=possible_conditions, question=question, possible_sql=possible_sql, exec_err=exec_err, few_shot_examples=sql_generation_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions) 
        # print("refinement_prompt: \n", prompt)
        return prompt
    
    def construct_filtering_prompt(self, db_path: str, db_id: str, question: str, evidence: str, schema_dict: Dict, db_descriptions: str)->str:
        """
        The function constructs the prompt required for the database schema filtering stage

        Arguments:  
            db_path (str): The database sqlite file path.
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            prompt (str): prompt for database schema filtering stage
        """
        schema_filtering_prompt_template_path =  os.path.join(os.getcwd(), "prompt_templates/schema_filter_prompt_template.txt")
        with open(schema_filtering_prompt_template_path, 'r') as f:
            schema_filtering_template = f.read()

        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        schema_filtering_few_shot_examples = schema_filtering_few_shot_prep(few_shot_data_path, q_db_id=db_id, level_shot_number=self.elsn, schema_existance=self.efsse, mode=self.mode)
        db_samples = extract_db_samples_enriched_bm25(question, evidence, db_path=db_path, schema_dict=schema_dict, sample_limit=self.db_sample_limit)
        schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)
        prompt = fill_prompt_template(template=schema_filtering_template, schema=schema, db_samples=db_samples, question=question, few_shot_examples=schema_filtering_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions)
        # print("\nSchema Filtering Prompt: \n", prompt)
    
        return prompt

    
    def candidate_sql_generation_module(self, db_path: str, db_id: int, question: str, evidence: str, filtered_schema_dict: Dict, db_descriptions: str):
        """
        This function generates candidate SQL for answering the question.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_candidate_sql_generation_prompt(db_path=db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="candidate_sql_generation", prompt=prompt, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=self.n, provider=self.provider)
        try:
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            return response_object
        return response_object

    
    def sql_refinement_module(self, db_path: str, db_id: int, question: str, evidence: str, possible_sql: str, exec_err: str, filtered_schema_dict: Dict, db_descriptions: str):
        """
        This function refines or re-generates a SQL query for answering the question.
        Possible SQL query, possible conditions generated from possible SQL query and execution error if it is exist are leveraged for better SQL refinement.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            possible_sql (str): Previously generated possible SQL query for the question
            exec_err (str): Taken execution error when possible SQL is executed 
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_sql_refinement_prompt(db_path=db_path, db_id=db_id, question=question, evidence=evidence, possible_sql=possible_sql, exec_err=exec_err, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="sql_refinement", prompt=prompt, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=self.n, provider=self.provider)
        try:
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            return response_object
        return response_object
    

    def schema_filtering_module(self, db_path: str, db_id: str, question: str, evidence: str, schema_dict: Dict, db_descriptions: str):
        """
        The function filters the database schema by eliminating the unnecessary tables and columns

        Arguments:  
            db_path (str): The database sqlite file path.
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_filtering_prompt(db_path=db_path, db_id=db_id, question=question, evidence=evidence, schema_dict=schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="schema_filtering", prompt=prompt, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=self.n)
        try:
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            return response_object

        return response_object

    # ─────────────────────────────────────────────────────────────────────────────
    # NEW: QE Combination Module
    # ─────────────────────────────────────────────────────────────────────────────
    def construct_qe_combination_prompt(
        self,
        db_path: str,
        q_id: int,
        db_id: str,
        question: str,
        evidence: str,
        possible_conditions: str,
        schema_dict: Dict,
        db_descriptions: str,
        enriched_questions_with_reasoning: List[Dict[str, str]],
    ) -> str:
        """
        Constructs the prompt for the QE combination stage (multi-enriched → one combined).
        Mirrors QE: same template loading, same few-shot source, same section building.
        """
        combination_template_path = os.path.join(os.getcwd(), "prompt_templates/qe_combination_prompt_template.txt")
        qe_combination_prompt_template = extract_qe_combination_prompt_template(combination_template_path)

        # Use the SAME few-shot file and logic as Question Enrichment
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        comb_few_shot_examples = qe_combination_few_shot_prep(
            few_shot_data_path=few_shot_data_path,
            q_id=q_id,
            q_db_id=db_id,
            level_shot_number=self.elsn,
            schema_existance=self.efsse,
            enrichment_level=self.enrichment_level,  # "basic" or "complex" to match QE
            mode=self.mode,
        )

        # Same sample and schema prep as QE
        db_samples = extract_db_samples_enriched_bm25(
            question, evidence, db_path=db_path, schema_dict=schema_dict, sample_limit=self.db_sample_limit
        )
        schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)

        prompt = fill_qe_combination_prompt_template(
            template=qe_combination_prompt_template,
            schema=schema,
            db_samples=db_samples,
            question=question,
            possible_conditions=possible_conditions,
            few_shot_examples=comb_few_shot_examples,
            evidence=evidence,
            db_descriptions=db_descriptions,
            enriched_question_list_with_reasoning=enriched_questions_with_reasoning,
        )
        return prompt

    def qe_combination_module(
        self,
        db_path: str,
        q_id: int,
        db_id: str,
        question: str,
        evidence: str,
        possible_conditions: str,
        schema_dict: Dict,
        db_descriptions: str,
        enriched_questions_with_reasoning: List[Dict[str, str]],
    ) -> Dict:
        """
        Calls the LLM to combine multiple enriched questions into ONE.
        Returns a response_object consistent with other modules (JSON with 'combined_enriched_question').
        """
        prompt = self.construct_qe_combination_prompt(
            db_path=db_path,
            q_id=q_id,
            db_id=db_id,
            question=question,
            evidence=evidence,
            possible_conditions=possible_conditions,
            schema_dict=schema_dict,
            db_descriptions=db_descriptions,
            enriched_questions_with_reasoning=enriched_questions_with_reasoning,
        )
        response_object = create_response(
            stage="qe_combination",
            prompt=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            provider=self.provider
        )
        try:
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            return response_object
        return response_object


    def construct_iterative_enricher_prompt(
        self,
        *,
        db_path: str,
        q_id: int,
        db_id: str,
        question: str,
        enriched_question_v1: str,
        enrichment_reasoning_v1: str,
        evidence: str,
        possible_conditions: str,
        schema_dict: dict,
        db_descriptions: str,
    ) -> str:
        template_path = os.path.join(os.getcwd(), "prompt_templates/iterative_enricher_prompt_template.txt")
        template = extract_iterative_enricher_prompt_template(template_path)

        # few-shots taken from the same pool as QE, but using enrichment_level="complex" to surface v2 examples
        few_shot_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        fewshots = iterative_enricher_few_shot_prep(
            few_shot_data_path=few_shot_path,
            q_id=q_id,
            q_db_id=db_id,
            level_shot_number=self.elsn,
            schema_existance=self.efsse,
            enrichment_level=self.enrichment_level,
            mode=self.mode,
        )

        # samples and schema consistent with QE
        db_samples = extract_db_samples_enriched_bm25(
            question,
            evidence,
            db_path=db_path,
            schema_dict=schema_dict,
            sample_limit=self.db_sample_limit
        )
        schema_text = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)

        prompt = fill_iterative_enricher_prompt_template(
            template=template,
            schema=schema_text,
            db_samples=db_samples,
            question=question,
            enriched_question_v1=enriched_question_v1,
            enrichment_reasoning_v1=enrichment_reasoning_v1,
            possible_conditions=possible_conditions,
            few_shot_examples=fewshots,
            evidence=evidence,
            db_descriptions=db_descriptions
        )
        return prompt

    def iterative_enricher_module(
        self,
        *,
        db_path: str,
        q_id: int,
        db_id: str,
        question: str,
        enriched_question_v1: str,
        enrichment_reasoning_v1: str,
        evidence: str,
        possible_conditions: str,
        schema_dict: dict,
        db_descriptions: str,
    ):
        prompt = self.construct_iterative_enricher_prompt(
            db_path=db_path,
            q_id=q_id,
            db_id=db_id,
            question=question,
            enriched_question_v1=enriched_question_v1,
            enrichment_reasoning_v1=enrichment_reasoning_v1,
            evidence=evidence,
            possible_conditions=possible_conditions,
            schema_dict=schema_dict,
            db_descriptions=db_descriptions
        )
        resp = create_response(
            stage="iterative_enricher",
            prompt=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            provider=self.provider
        )
        try:
            resp = self.convert_message_content_to_dict(resp)
        except:
            return resp
        return resp

    def construct_extract_keywords_prompt(self, *, question: str, hint: str) -> str:
        """
        Build the Extract Keywords prompt. Template is loaded here (pipeline level),
        same style as other stages.
        """
        kw_template_path = os.path.join(os.getcwd(), "prompt_templates/extract_keywords_prompt_template.txt")
        kw_template = extract_extract_keyword_prompt_template(kw_template_path)
        prompt = fill_extract_keywords_prompt_template(
            template=kw_template,
            question=question,
            hint=hint,
        )
        return prompt

    def extract_keywords_module(self, *, question: str, hint: str):
        """
        Runs the Extract Keywords stage.
        Returns a dict with parsed keywords and usage, ready to be embedded into t2s_object.
        """
        prompt = self.construct_extract_keywords_prompt(question=question, hint=hint)
        resp = create_response(
            stage="extract_keywords",
            prompt=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            provider=self.provider
        )
        try:
            resp = self.convert_message_content_to_dict(resp)
        except:
            return resp
        return resp


    def filter_columns_module(self, *, question: str, hint: str, db_path: str) -> Dict[str, List[str]]:
        """
        Runs the Filter Columns stage.
        Returns a dict with tables as keys and lists of relevant column names as values.
        """
        # Retrieve database schema information and sample values
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        col_type_map = {}
        fk_map = {}
        example_map = {}
        for table_name in tables:
            cursor.execute(f"PRAGMA table_info(`{table_name}`);")
            cols_info = cursor.fetchall()
            col_type_map[table_name] = {}
            example_map[table_name] = {}
            col_names = [col[1] for col in cols_info]
            # Fetch one sample row from table for example values
            row = None
            try:
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1;")
                row = cursor.fetchone()
            except Exception as e:
                logging.error(f"Error fetching sample row from table '{table_name}': {e}")
            for col in cols_info:
                col_name = col[1]
                col_type = col[2] or "UNKNOWN"
                col_type_map[table_name][col_name] = col_type
                example_val = None
                if row:
                    cid = col[0]
                    if cid < len(row):
                        example_val = row[cid]
                example_map[table_name][col_name] = example_val
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
            fks = cursor.fetchall()
            fk_map[table_name] = []
            for fk in fks:
                # PRAGMA returns (id, seq, table, to, from, on_update, on_delete, match)
                ref_table = fk[2]
                ref_col = fk[3]
                from_col = fk[4]
                fk_map[table_name].append((from_col, ref_table, ref_col))
        conn.close()
        # Load column descriptions (if available) from JSON
        description_map = {}
        try:
            bird_sql_path = os.getenv('BIRD_DB_PATH', self.dataset_path)
            db_dir = os.path.dirname(db_path)
            col_meaning_path = os.path.join(bird_sql_path, self.mode, "column_meaning.json")
            with open(col_meaning_path, 'r') as f:
                all_meanings = json.load(f)
            db_id = os.path.basename(db_dir)
            if db_id in all_meanings:
                description_map = all_meanings[db_id]
        except Exception as e:
            logging.error(f"Column meaning file not found or error loading: {e}")
            description_map = {}
        # Filter columns using LLM
        filtered_schema = {table: [] for table in tables}
        template_path = os.path.join(os.getcwd(), "prompt_templates/filter_columns_prompt_template.txt")
        with open(template_path, 'r') as f:
            template_text = f.read()
        for table in tables:
            for column in col_type_map[table]:
                col_type = col_type_map[table][column]
                descr = description_map.get(column, "")
                if not descr:
                    descr = "No description available"
                example_val = example_map[table].get(column)
                example_str = f"`{example_val}`" if example_val is not None else "`N/A`"
                column_profile = (f"Table name: `{table}`\n"
                                  f"Original column name: `{column}`\n"
                                  f"Data type: {col_type}\n"
                                  f"Description: {descr}\n"
                                  f"Example of values in the column: {example_str}")
                prompt = template_text.format(QUESTION=question, HINT=(hint or "No hint"), COLUMN_PROFILE=column_profile)
                try:
                    response_object = create_response(stage="filter_columns", prompt=prompt, model=self.model,
                                                      max_tokens=self.max_tokens, temperature=self.temperature,
                                                      top_p=self.top_p, n=1, provider=self.provider)
                except Exception as e:
                    logging.error(f"LLM call failed for column filtering on {table}.{column}: {e}")
                    continue
                raw = response_object.choices[0].message.content
                try:
                    if isinstance(raw, str):
                        content = json.loads(raw)
                    elif isinstance(raw, dict):
                        content = raw
                    else:
                        raise TypeError(f"Unexpected content type: {type(raw)}")
                except Exception as e:
                    logging.error(
                        f"Invalid JSON in filter_columns_module for {table}.{column}: {e}"
                    )
                    continue

                val = str(content.get("is_column_information_relevant", "")).lower()
                if val.startswith("y"):
                    filtered_schema[table].append(column)
        # Add bridging tables that connect at least two relevant tables
        relevant_tables = [t for t, cols in filtered_schema.items() if cols]
        for table in tables:
            if table not in relevant_tables:
                ref_tables = {fk[1] for fk in fk_map.get(table, []) if fk[1] in relevant_tables}
                if len(ref_tables) >= 2:
                    filtered_schema[table] = [fk[0] for fk in fk_map[table]]
                    relevant_tables.append(table)
        # Store schema info for later stages
        self.col_type_map = col_type_map
        self.fk_map = fk_map
        return filtered_schema


    def select_tables_module(self, *, question: str, hint: str, filtered_schema: Dict[str, List[str]]) -> List[str]:
        """
        Runs the Select Tables stage.
        Returns a list of table names needed to answer the question.
        """
        schema_lines = []
        for table, cols in filtered_schema.items():
            if cols:
                schema_lines.append(f"Table: {table}")
                for col in cols:
                    col_type = self.col_type_map.get(table, {}).get(col, "UNKNOWN")
                    extra = ""
                    fk_info = [fk for fk in self.fk_map.get(table, []) if fk[0] == col]
                    if fk_info:
                        ref_table, ref_col = fk_info[0][1], fk_info[0][2]
                        extra = f"(FK -> {ref_table}.{ref_col})"
                    line = f"- {col} ({col_type})"
                    if extra:
                        line += " " + extra
                    if hasattr(self, 'similar_examples_map'):
                        val = self.similar_examples_map.get((table, col))
                        if val is not None:
                            line += f" -- examples: `{val}`"
                    schema_lines.append(line)
        schema_overview = "\n".join(schema_lines)
        template_path = os.path.join(os.getcwd(), "prompt_templates/select_tables_prompt_template.txt")
        with open(template_path, 'r') as f:
            template_text = f.read()
        prompt = template_text.format(DATABASE_SCHEMA=schema_overview, QUESTION=question, HINT=(hint or "No hint"))
        try:
            response_object = create_response(stage="select_tables", prompt=prompt, model=self.model,
                                              max_tokens=self.max_tokens, temperature=self.temperature,
                                              top_p=self.top_p, n=1, provider=self.provider)
        except Exception as e:
            logging.error(f"LLM call failed for select_tables_module: {e}")
            return []
        selected_tables = []
        raw = response_object.choices[0].message.content
        try:
            if isinstance(raw, str):
                content = json.loads(raw)
            elif isinstance(raw, dict):
                content = raw
            else:
                raise TypeError(f"Unexpected content type: {type(raw)}")

            if isinstance(content, dict) and "table_names" in content:
                selected_tables = content["table_names"]
        except Exception as e:
            logging.error(f"Invalid JSON in select_tables_module: {e}")

        return selected_tables

    def select_columns_module(self, *, question: str, hint: str, selected_tables: List[str], filtered_schema: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Runs the Select Columns stage.
        Returns a dict of table names to lists of column names needed for the SQL query.
        """
        tentative_schema = {t: list(filtered_schema.get(t, [])) for t in selected_tables}
        if not tentative_schema:
            return {}
        for table in selected_tables:
            for fk in self.fk_map.get(table, []):
                from_col, ref_table, ref_col = fk
                if ref_table in selected_tables:
                    if from_col not in tentative_schema[table]:
                        tentative_schema[table].append(from_col)
                    if ref_col not in tentative_schema.get(ref_table, []):
                        tentative_schema.setdefault(ref_table, []).append(ref_col)
        schema_lines = []
        for table, cols in tentative_schema.items():
            if cols:
                schema_lines.append(f"Table: {table}")
                for col in cols:
                    col_type = self.col_type_map.get(table, {}).get(col, "UNKNOWN")
                    extra = ""
                    fk_info = [fk for fk in self.fk_map.get(table, []) if fk[0] == col]
                    if fk_info:
                        ref_table, ref_col = fk_info[0][1], fk_info[0][2]
                        extra = f"(FK -> {ref_table}.{ref_col})"
                    line = f"- {col} ({col_type})"
                    if extra:
                        line += " " + extra
                    if hasattr(self, 'similar_examples_map'):
                        val = self.similar_examples_map.get((table, col))
                        if val is not None:
                            line += f" -- examples: `{val}`"
                    schema_lines.append(line)
        schema_overview = "\n".join(schema_lines)
        template_path = os.path.join(os.getcwd(), "prompt_templates/select_columns_prompt_template.txt")
        with open(template_path, 'r') as f:
            template_text = f.read()
        prompt = template_text.format(DATABASE_SCHEMA=schema_overview, QUESTION=question, HINT=(hint or "No hint"))
        try:
            response_object = create_response(stage="select_columns", prompt=prompt, model=self.model,
                                              max_tokens=self.max_tokens, temperature=self.temperature,
                                              top_p=self.top_p, n=(self.n if hasattr(self, 'n') else 1),
                                              provider=self.provider)
        except Exception as e:
            logging.error(f"LLM call failed for select_columns_module: {e}")
            return {}
        final_selected_columns = {}
        raw = response_object.choices[0].message.content
        try:
            if isinstance(raw, str):
                content = json.loads(raw)
            elif isinstance(raw, dict):
                content = raw
            else:
                raise TypeError(f"Unexpected content type: {type(raw)}")
        except Exception as e:
            logging.error(f"Invalid JSON in select_columns_module: {e}")
            return {}

        if isinstance(content, dict):
            # unwrap if model nested under 'selected_columns'
            if "selected_columns" in content and isinstance(content["selected_columns"], dict):
                inner = content["selected_columns"]
            else:
                inner = content

            inner.pop("chain_of_thought_reasoning", None)
            for table, cols in inner.items():
                if table == "chain_of_thought_reasoning":
                    continue
                final_selected_columns[table] = cols
        for table in list(final_selected_columns.keys()):
            if table not in selected_tables or not final_selected_columns[table]:
                final_selected_columns.pop(table, None)
        return final_selected_columns






    
    
