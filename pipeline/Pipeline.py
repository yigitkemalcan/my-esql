import os
import json
from utils.prompt_utils import *
from utils.db_utils import * 
from utils.openai_utils import create_response
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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


        ### STAGE 1: FILTERING THE DATABASE SCHEMA
        # -- original question is used.
        # -- Original Schema is used.
        schema_filtering_response_obj = self.schema_filtering_module(db_path=db_path, db_id=db_id, question=question, evidence=evidence, schema_dict=original_schema_dict, db_descriptions=db_descriptions)
        # print("schema_filtering_response_obj: \n", schema_filtering_response_obj)
        try:
            t2s_object["schema_filtering"] = {
                "filtering_reasoning": schema_filtering_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "filtered_schema_dict": schema_filtering_response_obj.choices[0].message.content['tables_and_columns'],
                "prompt_tokens": schema_filtering_response_obj.usage.prompt_tokens,
                "completion_tokens": schema_filtering_response_obj.usage.completion_tokens,
                "total_tokens": schema_filtering_response_obj.usage.total_tokens,
            }
        except Exception as e:
            logging.error(f"Error in reaching content from schema filtering response for question_id {q_id}: {e}")
            t2s_object["schema_filtering"] = f"{e}"
            return t2s_object

        ### STAGE 1.1: FILTERED SCHEMA CORRECTION
        filtered_schema_dict = schema_filtering_response_obj.choices[0].message.content['tables_and_columns']
        filtered_schema_dict, filtered_schema_problems = filtered_schema_correction(db_path=db_path, filtered_schema_dict=filtered_schema_dict) 
        t2s_object["schema_filtering_correction"] = {
            "filtered_schema_problems": filtered_schema_problems,
            "final_filtered_schema_dict": filtered_schema_dict
        }

        schema_statement = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        t2s_object["create_table_statement"] = schema_statement

        ### STAGE 2: Candidate SQL GENERATION
        # -- Original question is used
        # -- Filtered Schema is used 
        sql_generation_response_obj =  self.candidate_sql_generation_module(db_path=db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
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
        q_enrich_response_obj = self.question_enrichment_module(db_path=db_path, q_id=q_id, db_id=db_id, question=question, evidence=evidence, possible_conditions=possible_conditions, schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
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
        sql_generation_response_obj =  self.sql_refinement_module(db_path=db_path, db_id=db_id, question=enriched_question, evidence=evidence, possible_sql=possible_sql, exec_err=exec_err, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
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




    
    
