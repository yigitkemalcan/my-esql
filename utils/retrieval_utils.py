import os
import json
import string
import logging
import difflib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List
import pickle
from datasketch import MinHash, MinHashLSH
from vertexai.language_models import TextEmbeddingModel
from vertexai import init as vertex_init
from pathlib import Path
from utils.db_utils import execute_sql
from openai import OpenAI

# Global flag to track if Vertex AI has been initialized
_vertex_initialized = False

def _ensure_vertex_init():
    """Ensure Vertex AI is initialized before using embedding models."""
    global _vertex_initialized
    if _vertex_initialized:
        return
    
    project = os.getenv("GCP_PROJECT")
    if not project:
        raise EnvironmentError("GCP_PROJECT environment variable must be set for Vertex AI embeddings")
    
    # Try multiple locations for embedding API
    locations = ["us-central1", "us-east5", "europe-west1", "asia-southeast1"]
    env_loc = os.getenv("GCP_LOCATION")
    if env_loc:
        locations.insert(0, env_loc)
    
    last_error = None
    for location in locations:
        try:
            vertex_init(project=project, location=location)
            logging.info(f"Vertex AI initialized successfully with location: {location}")
            _vertex_initialized = True
            return
        except Exception as e:
            last_error = e
            logging.warning(f"Failed to initialize Vertex AI in {location}: {e}")
            continue
    
    raise RuntimeError(f"Could not initialize Vertex AI in any location. Last error: {last_error}")

def _embed_with_openai(text_list):
    """Fallback embedding using OpenAI API."""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Set API key in environment (OpenAI client reads from env by default)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize client without arguments to avoid compatibility issues
        # The client will automatically read OPENAI_API_KEY from environment
        client = OpenAI()
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_list
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logging.error(f"OpenAI embedding failed: {e}")
        raise

def embed_batch(text_list):
    """
    Embed a batch of text using best available embedding service.
    
    Priority:
    1. OpenAI (more reliable, higher quota)
    2. Vertex AI (if OpenAI unavailable)
    """
    
    # Try OpenAI first (more reliable)
    if os.getenv("OPENAI_API_KEY"):
        try:
            embeddings = _embed_with_openai(text_list)
            logging.info("Successfully used OpenAI embeddings")
            return embeddings
        except Exception as e:
            logging.warning(f"OpenAI embeddings failed, trying Vertex AI: {e}")
    
    # Fallback to Vertex AI
    try:
        _ensure_vertex_init()
    except Exception as e:
        logging.error(f"Cannot initialize Vertex AI and OpenAI not available: {e}")
        raise
    
    model_names = [
        "text-embedding-004",
        "text-embedding-005", 
        "textembedding-gecko@002",
        "textembedding-gecko@001",
        "textembedding-gecko"
    ]
    
    last_error = None
    for model_name in model_names:
        try:
            model = TextEmbeddingModel.from_pretrained(model_name)
            embeddings = model.get_embeddings(text_list)
            logging.info(f"Successfully used Vertex AI model: {model_name}")
            return [e.values for e in embeddings]
        except Exception as e:
            last_error = e
            logging.warning(f"Failed to use model {model_name}: {e}")
            continue
    
    # If all models fail, raise the last error
    raise Exception(f"All embedding services failed. Last error: {last_error}")

def embed_text(text: str):
    return embed_batch([text])[0]


def nltk_downloads():
    nltk.download('stopwords') # Download the stopwords
    nltk.download('punkt')  # Download the punkt tokenizer
    nltk.download('punkt_tab') # Download the punkt_tab
    return


def save_dataframe_to_csv(df: pd.DataFrame, path: str):
    """
    Saves the given pandas DataFrame to a CSV file at the specified path.

    Arguments:
    df (pd.DataFrame): The DataFrame to save.
    path (str): The file path where the CSV file will be saved.
    """
    try:
        df.to_csv(path, index=False)  # Set index=False to avoid saving the index column
        print(f"DataFrame saved successfully to {path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the DataFrame: {e}")


def clean_text(textData: str)-> str:
    """
    The function process the given textData by removing stop words, removing punctuation marks and lowercasing the textData.

    Arguments:
        textData (str): text to be cleaned
    
    Returns:
        processedTextData (str): cleaned text
    """

    if isinstance(textData, str):
        textData = textData.lower()
        textData = textData.replace("       ", '')

        # Removing punctuations
        # textData = textData.translate(str.maketrans('', '', string.punctuation)) # converts "don't" to "dont"
        textData = textData.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) # converts "don't" to "don t"

        # Removing stopwords
        stopWordsSet = set(stopwords.words('english'))
        tokens = word_tokenize(textData)
        tokens = [token for token in tokens if not token.lower() in stopWordsSet]

        processedTextData = ' '.join(tokens)
        return processedTextData
    else:
    # if the text data is NaN return empty string
        return ''
    

def construct_column_information(table_desc_df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    The function combines the original column name, column description, data format, and value description information from table description CSV files into a single descriptive string for each column and adds it as a new column in the DataFrame.

    Arguments:
        table_desc_df (pd.DataFrame): DataFrame containing table descriptions.
        table_name (str): Name of the table.

    Returns:
        pd.Series: constructed single text column information for each column 
    """
    # Function to build column info for each row
    def build_column_info(row):
        column_info = f"The information about the {row['original_column_name']} column of the {table_name} table [{table_name}.{row['original_column_name']}] is as following."

        if pd.notna(row['column_description']):
            column_info += f" The {row['original_column_name']} column can be described as {row['column_description']}."
        if pd.notna(row['value_description']):
            column_info += f" The value description for the {row['original_column_name']} is {row['value_description']}"
        
        column_info = column_info.replace("       ", ' ')
        column_info = column_info.replace("       ", ' ')
        return column_info

    # Apply the function to create the "column_info" column
    # table_desc_df['column_info'] = table_desc_df.apply(build_column_info, axis=1)
    column_info_series = table_desc_df.apply(build_column_info, axis=1)

    return column_info_series


def process_database_descriptions(database_description_path: str):
    """
    Processes multiple CSV files in the given directory, applies the pre-existing construct_column_information function to each,
    and combines the "column_info" columns into a single DataFrame which is then saved as db_description.csv.

    Arguments:
        database_description_path (str): Path to the directory containing database description CSV files.
    """

    # List to store column_info from each file
    all_column_infos = []

    # Iterate over each file in the directory
    for filename in os.listdir(database_description_path):
        if filename.endswith(".csv") and filename != "db_description.csv" :
            print(f"------> {filename} table start to be processed.")
            file_path = os.path.join(database_description_path, filename)
            try:
                df = pd.read_csv(file_path)
            except:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                
            table_name = filename.replace('.csv', '')
            column_info_series = construct_column_information(df, table_name)
            # Convert the Series to a DataFrame with a single column named 'column_info'
            column_info_df = column_info_series.to_frame(name='column_info')
            all_column_infos.append(column_info_df)

    # Combine all column_info data into a single DataFrame
    all_info_df = pd.concat(all_column_infos, ignore_index=True)

    # Save the DataFrame to a CSV file
    output_path = os.path.join(database_description_path, 'db_description.csv')
    all_info_df.to_csv(output_path, index=False)
    print(f"---> Database information saved successfully to {output_path}")
    
    return



def process_all_dbs(dataset_path: str, mode: str):
    """
    The function processes description of all databases and construct db_description.csv file for all databases.

    Arguments:
        dataset_path (str): General dataset path
        mode (str): Either dev, test or train
    """
    nltk_downloads() # download nltk stop words
    databases_path = dataset_path + f"/{mode}/{mode}_databases"
   
    for db_directory in os.listdir(databases_path):
        #thank you, Apple inc. for creating .DS_Store files
        if db_directory == ".DS_Store":
            continue
        print(f"----------> Start to process {db_directory} database.")
        db_description_path = databases_path + "/" + db_directory + "/database_description"
        process_database_descriptions(database_description_path=db_description_path)

    print("\n\n All databases processed and db_description.csv files are created for all.\n\n")
    return


def get_relevant_db_descriptions(database_description_path: str, question: str, relevant_description_number: int = 6) -> List[str]:
    """
    CHESS-SQL style: retrieve most relevant schema descriptions via vector search
    instead of BM25. Falls back to building the vector DB if missing.
    """
    # ensure description CSV exists
    db_description_csv_path = os.path.join(database_description_path, "db_description.csv")
    if not os.path.exists(db_description_csv_path):
        process_database_descriptions(database_description_path)

    # ensure vector DB exists and load it
    db_path = Path(database_description_path).parent
    db_id = db_path.name
    vec_dir = db_path / "context_vector_db"
    vectors_file = vec_dir / f"{db_id}_vectors.npy"
    meta_file = vec_dir / f"{db_id}_metadata.pkl"
    if not vectors_file.exists() or not meta_file.exists():
        logging.warning(f"Vector DB not found for {db_id}. Building now...")
        make_db_context_vec_db(db_path)

    vectors = np.load(vectors_file)
    with open(meta_file, "rb") as f:
        docs_meta = pickle.load(f)

    # embed query
    query_vec = np.array(embed_text(question), dtype=float)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        q_norm = 1e-9
    query_vec = query_vec / q_norm

    sims = np.dot(vectors, query_vec)
    total_candidates = len(sims)
    if total_candidates == 0:
        logging.warning(f"No description vectors found for {db_id} at {database_description_path}")
        return []
    if relevant_description_number <= 0:
        return []
    k = min(relevant_description_number, total_candidates)
    # use a positive kth index to stay within bounds
    kth = total_candidates - k
    top_idx = np.argpartition(sims, kth)[kth:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    relevant_db_descriptions = []
    for idx in top_idx:
        meta = docs_meta[idx]
        text = meta.get("source_text") or meta.get("column_description") or meta.get("value_description", "")
        if text and text not in relevant_db_descriptions:
            relevant_db_descriptions.append(text)
    return relevant_db_descriptions

def get_db_column_meanings(database_column_meaning_path: str, db_id: str) -> List[str]:
    """
    The function extracts required database column meanings.

    Arguments:
        database_column_meaning_path (str): path to the column_meaning.json
        db_id (str): name of the database whose columns' meanings will be extracted

    Returns:
        List[str]: A list of strings explaining the database column meanings.
    """
    # Load the JSON file
    with open(database_column_meaning_path, 'r') as file:
        column_meanings = json.load(file)
    
    # Initialize a list to store the extracted meanings
    meanings = []

    # Iterate through each key in the JSON
    for key, explanation in column_meanings.items():
        # Check if the key starts with the given db_id
        if key.startswith(db_id + "|"):
            # Extract the table name and column name from the key
            _, table_name, column_name = key.split("|")
            # Construct the meaning string in the desired format
            meaning = f"# Meaning of {column_name} column of {table_name} table in database is that {explanation.strip('# ').strip()}"
            meanings.append(meaning)
    
    return meanings

def load_tables_description(database_description_path: str, use_value_description: bool = True):
    """Load table and column descriptions from all CSV files in the given directory."""
    tables_desc = {}
    desc_path = Path(database_description_path)
    if not desc_path.exists():
        logging.warning(f"Description path does not exist: {desc_path}")
        return tables_desc
    # Iterate CSV files
    for csv_file in desc_path.glob("*.csv"):
        if csv_file.name == "db_description.csv":
            continue
        table = csv_file.stem.lower().strip()
        tables_desc[table] = {}
        try:
            df = pd.read_csv(csv_file, index_col=False, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(csv_file, index_col=False, encoding='cp1252')
        for _, row in df.iterrows():
            col_name = str(row['original_column_name']).strip()
            expanded_name = str(row.get('column_name', '')).strip() if pd.notna(row.get('column_name')) else ""
            col_desc = str(row.get('column_description', '')).replace('\n', ' ').replace("commonsense evidence:", "").strip() if pd.notna(row.get('column_description')) else ""
            val_desc = ""
            if use_value_description and pd.notna(row.get('value_description')):
                val_desc = str(row['value_description']).replace('\n', ' ').replace("commonsense evidence:", "").strip()
                if val_desc.lower().startswith("not useful"):
                    val_desc = val_desc[10:].strip()  # remove "not useful" prefix if present
            tables_desc[table][col_name.lower()] = {
                "original_column_name": col_name,
                "column_name": expanded_name,
                "column_description": col_desc,
                "value_description": val_desc
            }
    return tables_desc

def make_db_context_vec_db(db_directory_path: str, use_value_description: bool = True):
    """Creates a semantic vector index of the database schema (column descriptions) for the given database directory."""
    db_path = Path(db_directory_path)
    db_id = db_path.name
    logging.info(f"Creating context vector database for {db_id}")
    # Load table descriptions
    tables_desc = load_tables_description(db_path / "database_description", use_value_description)
    docs_texts = []
    docs_meta = []
    # Prepare texts for embedding
    for table, columns in tables_desc.items():
        for col_key, col_info in columns.items():
            metadata = {
                "table_name": table,
                "original_column_name": col_info["original_column_name"],
                "column_name": col_info["column_name"],
                "column_description": col_info["column_description"],
                "value_description": col_info["value_description"]
            }
            # Column description text
            if col_info["column_description"]:
                docs_texts.append(col_info["column_description"])
                docs_meta.append({**metadata, "source_text": col_info["column_description"]})
            # Value description text (if using)
            if use_value_description and col_info["value_description"]:
                docs_texts.append(col_info["value_description"])
                docs_meta.append({**metadata, "source_text": col_info["value_description"]})
    # Embed texts to vectors
    vectors = []
    if len(docs_texts) == 0:
        logging.warning(f"No descriptions to embed for {db_id}")
        return
    batch_size = 50
    for i in range(0, len(docs_texts), batch_size):
        batch = docs_texts[i:i+batch_size]
        try:
            batch_vectors = embed_batch(batch) 
        except Exception as e:
            logging.error(f"Embedding API call failed: {e}")
            raise
        # Extract embeddings
        vectors.extend(batch_vectors)
    vectors = np.array(vectors, dtype=float)
    # Normalize vectors for cosine similarity (optional)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    vectors_normalized = vectors / norms
    # Save the vector index
    vec_dir = db_path / "context_vector_db"
    if vec_dir.exists():
        # remove old index files
        import shutil
        shutil.rmtree(vec_dir)
    vec_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_dir / f"{db_id}_vectors.npy", vectors_normalized)
    with open(vec_dir / f"{db_id}_metadata.pkl", "wb") as f:
        pickle.dump(docs_meta, f)
    logging.info(f"Vector index created with {len(vectors_normalized)} vectors at {vec_dir}")

def query_vector_db(database_description_path: str, query: str, top_k: int = 5):
    """Query the precomputed vector database for the most relevant schema descriptions."""
    db_path = Path(database_description_path).parent  # get parent directory of "database_description"
    db_id = db_path.name
    vec_dir = db_path / "context_vector_db"
    vectors_file = vec_dir / f"{db_id}_vectors.npy"
    meta_file = vec_dir / f"{db_id}_metadata.pkl"
    if not vectors_file.exists() or not meta_file.exists():
        # If vectors not precomputed, we can call make_db_context_vec_db on the fly (or warn)
        logging.warning(f"Vector DB not found for {db_id}. Building now...")
        make_db_context_vec_db(db_path)
    # Load vectors and metadata
    vectors = np.load(vectors_file)
    with open(meta_file, "rb") as f:
        docs_meta = pickle.load(f)
    # Embed the query
    try:
        query_vec = np.array(embed_text(query), dtype=float)
    except Exception as e:
        logging.error(f"Failed to embed query: {e}")
        raise
    # Normalize query vector
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        q_norm = 1e-9
    query_vec /= q_norm
    # Compute cosine similarity scores
    sims = np.dot(vectors, query_vec)  # assuming vectors are normalized
    # Get top_k indices
    top_idx = np.argpartition(sims, -top_k)[-top_k:]
    # Sort them by similarity
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    top_results = {}
    for idx in top_idx:
        meta = docs_meta[idx]
        table = meta["table_name"]
        col = meta["original_column_name"]
        # Prepare output entry
        if table not in top_results:
            top_results[table] = {}
        if col not in top_results[table]:
            top_results[table][col] = {
                "column_name": meta["column_name"],
                "column_description": meta["column_description"],
                "value_description": meta["value_description"],
                "score": float(sims[idx])
            }
    return top_results

def _create_minhash(signature_size: int, string: str, n_gram: int) -> MinHash:
    m = MinHash(num_perm=signature_size)
    # lower-case the string for uniformity
    s = str(string).lower()
    # if string is shorter than n_gram, just use itself
    if len(s) < n_gram:
        m.update(s.encode('utf8'))
    else:
        for i in range(len(s) - n_gram + 1):
            shingle = s[i:i+n_gram]
            m.update(shingle.encode('utf8'))
    return m

def make_db_lsh(db_directory_path: str, signature_size: int = 100, n_gram: int = 3, threshold: float = 0.2):
    """Creates a MinHash LSH index for the values in the given database."""
    db_path = Path(db_directory_path)
    db_id = db_path.name
    logging.info(f"Building LSH index for database {db_id}")
    # Get unique values from the DB
    sqlite_file = db_path / f"{db_id}.sqlite"
    if not sqlite_file.exists():
        # try to find any sqlite file in the directory
        candidates = list(db_path.glob("*.sqlite"))
        if not candidates:
            logging.error(f"No SQLite database file found in {db_path}")
            return
        sqlite_file = candidates[0]
    unique_vals = _get_unique_values(str(sqlite_file))
    # Save unique values (for record)
    prep_dir = db_path / "preprocessed"
    prep_dir.mkdir(exist_ok=True)
    with open(prep_dir / f"{db_id}_unique_values.pkl", "wb") as f:
        pickle.dump(unique_vals, f)
    # Create LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
    minhashes = {}
    # Insert MinHashes for each value
    for table, cols in unique_vals.items():
        for col, values in cols.items():
            for idx, val in enumerate(values):
                mh = _create_minhash(signature_size, val, n_gram)
                key = f"{table}_{col}_{idx}"
                lsh.insert(key, mh)
                minhashes[key] = (mh, table, col, val)
    # Save LSH index and minhashes
    with open(prep_dir / f"{db_id}_lsh.pkl", "wb") as f:
        pickle.dump(lsh, f)
    with open(prep_dir / f"{db_id}_minhashes.pkl", "wb") as f:
        pickle.dump(minhashes, f)
    logging.info(f"LSH index created for {db_id}: {len(minhashes)} values indexed.")

def _get_unique_values(db_path: str):
    """Retrieve unique text values for each text column in the database (excluding large or non-informative columns)."""
    unique_values = {}
    try:
        tables = [t[0] for t in execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';", fetch="all")]
    except Exception as e:
        logging.error(f"Failed to fetch tables from {db_path}: {e}")
        return unique_values
    # Identify primary keys to exclude
    primary_keys = []
    for table in tables:
        try:
            cols_info = execute_sql(db_path, f"PRAGMA table_info('{table}')", fetch="all")
        except Exception as e:
            logging.error(f"Could not get schema for table {table}: {e}")
            continue
        for col in cols_info:
            col_name = col[1]
            if col[5] == 1:  # PK flag in PRAGMA output
                primary_keys.append(col_name.lower())
    for table in tables:
        if table == "sqlite_sequence":
            continue
        unique_values[table] = {}
        # Get text-type columns excluding primary keys and certain keywords
        try:
            cols_info = execute_sql(db_path, f"PRAGMA table_info('{table}')", fetch="all")
        except Exception as e:
            logging.error(f"Schema fetch failed for {table}: {e}")
            continue
        text_columns = []
        for col in cols_info:
            name, col_type = col[1], col[2]
            name_lower = name.lower()
            if name_lower in primary_keys: 
                continue
            # consider as text if type contains "CHAR" or "TEXT" (or no type given which might allow text)
            if "char" in col_type.lower() or "text" in col_type.lower() or col_type == "":
                text_columns.append(name)
        for column in text_columns:
            col_lower = column.lower()
            # Skip columns that likely contain URLs, IDs, or non-semantic data
            if any(kw in col_lower for kw in ["_id", " uid", "url", "email", "phone", "address"]) or col_lower.endswith("id"):
                continue
            # Check distinct count and total length
            query = f"""
                SELECT SUM(LENGTH(val)), COUNT(val) FROM (
                    SELECT DISTINCT `{column}` as val FROM `{table}` 
                    WHERE `{column}` IS NOT NULL
                ) sub;
            """
            try:
                sum_len, count_dist = execute_sql(db_path, query, fetch="one")
            except Exception:
                sum_len, count_dist = None, 0
            if sum_len is None or count_dist == 0:
                continue
            avg_len = sum_len / count_dist if count_dist else 0
            # Apply CHESS conditions for skipping large columns
            if not (("name" in col_lower and sum_len < 5000000) or (sum_len < 2000000 and avg_len < 25) or count_dist < 100):
                # skip this column due to size
                logging.info(f"Skipping column {table}.{column} (distinct={count_dist}, total_len={sum_len})")
                continue
            # Fetch distinct values for this column
            try:
                values = execute_sql(db_path, f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL", fetch="all")
                values = [str(v[0]) for v in values]
            except Exception as e:
                logging.error(f"Error fetching values for {table}.{column}: {e}")
                values = []
            logging.info(f"{table}.{column}: {len(values)} distinct values")
            unique_values[table][column] = values
    return unique_values

# def query_db_values(db_directory_path: str, keywords: List[str], top_n: int = 3):
#     """Find columns with values similar to the given keywords using the precomputed LSH index."""
#     db_path = Path(db_directory_path)
#     db_id = db_path.name
#     prep_dir = db_path / "preprocessed"
#     lsh_file = prep_dir / f"{db_id}_lsh.pkl"
#     mh_file = prep_dir / f"{db_id}_minhashes.pkl"
#     if not lsh_file.exists() or not mh_file.exists():
#         logging.warning(f"LSH index not found for {db_id}. Building now...")
#         make_db_lsh(db_path)
#     # Load LSH and minhashes
#     with open(lsh_file, "rb") as f:
#         lsh = pickle.load(f)
#     with open(mh_file, "rb") as f:
#         minhashes = pickle.load(f)
#     similar_entities = {}  # {table: {column: [matching_value, ...]}}
#     for keyword in keywords:
#         if not keyword or keyword.strip() == "":
#             continue
#         mh = _create_minhash(100, keyword, 3)
#         results = lsh.query(mh)
#         # Compute actual Jaccard similarity for each result to rank
#         sims = []
#         for res in results:
#             mh2 = minhashes[res][0]
#             # Jaccard similarity approximation via MinHash
#             sim = mh.jaccard(mh2)
#             sims.append((res, sim))
#         sims.sort(key=lambda x: x[1], reverse=True)
#         top_hits = sims[:top_n]
#         for res, sim in top_hits:
#             _, table_name, col_name, value = minhashes[res]
#             table_dict = similar_entities.setdefault(table_name, {})
#             col_list = table_dict.setdefault(col_name, [])
#             # record the value (if not already recorded)
#             if value not in col_list:
#                 col_list.append(value)
#     return similar_entities

def query_db_values(db_directory_path: str, keywords: List[str], top_n: int = 10,
                    max_columns: int = 6, max_values_per_column: int = 5):
    """
    Find columns with values similar to the given keywords using the precomputed LSH index.
    Returns a pruned dict {table: {column: [values...]}}.
    """
    db_path = Path(db_directory_path)
    db_id = db_path.name
    prep_dir = db_path / "preprocessed"
    lsh_file = prep_dir / f"{db_id}_lsh.pkl"
    mh_file = prep_dir / f"{db_id}_minhashes.pkl"

    if not lsh_file.exists() or not mh_file.exists():
        logging.warning(f"LSH index not found for {db_id}. Building now...")
        make_db_lsh(db_path)

    with open(lsh_file, "rb") as f:
        lsh = pickle.load(f)
    with open(mh_file, "rb") as f:
        minhashes = pickle.load(f)

    # 1) Collect hits per (table, col, value) with scores
    column_scores = {}   # (table, col) -> aggregated_score
    value_hits = {}      # (table, col) -> list of (value, score)

    for keyword in keywords:
        if not keyword or keyword.strip() == "":
            continue

        mh = _create_minhash(100, keyword, 3)
        results = lsh.query(mh)

        sims = []
        for res in results:
            mh2 = minhashes[res][0]
            sim = mh.jaccard(mh2)
            sims.append((res, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        
        # Use only Jaccard + edit distance (skip expensive semantic embedding)
        for res, jac_sim in sims[:top_n]:
            _, table_name, col_name, value = minhashes[res]
            key = (table_name, col_name)

            # Skip if Jaccard similarity is too low (not a real match)
            if jac_sim < 0.3:
                continue

            # edit-distance similarity
            ed_sim = difflib.SequenceMatcher(None, keyword.lower(), str(value).lower()).ratio()

            # combined score: prioritize exact/substring matches
            combined_sim = 0.6 * jac_sim + 0.4 * ed_sim

            # aggregate score per column
            column_scores[key] = column_scores.get(key, 0.0) + combined_sim

            # store value-level hit
            vlist = value_hits.setdefault(key, [])
            vlist.append((value, combined_sim))

    if not column_scores:
        return {}

    # 2) Select top columns overall - be more strict about threshold
    sorted_cols = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Only keep columns with reasonable scores
    threshold = 0.4
    top_cols_filtered = [(k, score) for k, score in sorted_cols if score >= threshold][:max_columns]
    
    if not top_cols_filtered:
        # If threshold is too strict, fall back to top max_columns
        top_cols_filtered = sorted_cols[:max_columns]
    
    # 3) For each column, keep top values
    similar_entities = {}
    for (table_name, col_name), _ in top_cols_filtered:
        vals_scores = value_hits.get((table_name, col_name), [])
        vals_scores.sort(key=lambda x: x[1], reverse=True)
        top_vals = []
        seen = set()
        for v, _ in vals_scores:
            if v in seen:
                continue
            top_vals.append(v)
            seen.add(v)
            if len(top_vals) >= max_values_per_column:
                break

        tdict = similar_entities.setdefault(table_name, {})
        tdict[col_name] = top_vals

    return similar_entities


