"""Display and summarize Every Eval Ever dataset statistics."""

import argparse
import sys
from typing import List, Optional, Tuple

import duckdb
from huggingface_hub import HfFileSystem
from tqdm import tqdm

SEP = '=' * 60
SUB = '-' * 60

hffs = HfFileSystem()

repo_id = 'evaleval/EEE_datastore'
folder_path = 'viewer_parquets'

HUGGING_FACE_DATASTORE = f'datasets/{repo_id}/{folder_path}/**/*.parquet'


def execute_query(con, sql):
    return con.execute(sql).fetchall()


def section(title: str) -> None:
    print(f'\n{SEP}')
    print(f'  {title.upper()}')
    print(SUB)


def print_startup_summary(
    datastore: str,
    table_name: str,
    schema_urls: List[str],
    instance_urls: List[str],
) -> None:
    print(f'\n{SEP}')
    print('EVERY EVAL EVER STATS')
    print(SUB)
    print(f'datastore: {datastore}')
    print(f'tables:    {table_name}_schema, {table_name}_instances')
    print(f'schema files:   {len(schema_urls):,}')
    print(f'instance files: {len(instance_urls):,}')


def print_table_ready(name: str, row_count: int, file_count: int) -> None:
    print(
        f'\nLoaded {name}: {row_count:,} rows from {file_count:,} parquet files.\n'
    )


def get_parquet_columns(con, url: str) -> set[str]:
    columns = con.execute(
        'DESCRIBE SELECT * FROM read_parquet(?, filename=true)', [url]
    ).fetchall()
    return {column_name for column_name, *_ in columns}


def build_instance_select_sql(available_columns: set[str]) -> str:
    def scalar_column(name: str, sql_type: str) -> str:
        if name in available_columns:
            return f'CAST({name} AS {sql_type}) AS {name}'
        return f'CAST(NULL AS {sql_type}) AS {name}'

    def json_column(source_name: str, alias: str) -> str:
        if source_name in available_columns:
            return f'CAST(to_json({source_name}) AS JSON) AS {alias}'
        return f'CAST(NULL AS JSON) AS {alias}'

    select_columns = [
        scalar_column('schema_version', 'VARCHAR'),
        scalar_column('evaluation_id', 'VARCHAR'),
        scalar_column('model_id', 'VARCHAR'),
        scalar_column('evaluation_name', 'VARCHAR'),
        scalar_column('evaluation_result_id', 'VARCHAR'),
        scalar_column('sample_id', 'VARCHAR'),
        scalar_column('sample_hash', 'VARCHAR'),
        scalar_column('interaction_type', 'VARCHAR'),
        json_column('input', 'input'),
        json_column('output', 'output'),
        json_column('messages', 'messages'),
        json_column('answer_attribution', 'answer_attribution'),
        json_column('evaluation', 'evaluation'),
        json_column('token_usage', 'token_usage'),
        json_column('performance', 'performance'),
        scalar_column('error', 'VARCHAR'),
        json_column('metadata', 'metadata'),
        scalar_column('filename', 'VARCHAR'),
    ]

    return ',\n'.join(select_columns)


def read_data(datastore) -> Optional[Tuple[List[str], List[str]]]:
    files = hffs.glob(datastore)

    if not files:
        print('No parquet files found.')
        return None

    schema_urls = [f'hf://{f}' for f in files if f.endswith('dataset.parquet')]
    instance_urls = [
        f'hf://{f}' for f in files if f.endswith('dataset_samples.parquet')
    ]

    return schema_urls, instance_urls


def analyze_data(con, schema_table, instance_table) -> None:
    print(f'\n{SEP}')
    print('  EVERY EVAL EVER — DATASET ANALYSIS')
    print(f'  tables: {schema_table}, {instance_table}')
    print(SEP)

    # schema vs instance
    n_schema_files = execute_query(
        con, f'SELECT COUNT(DISTINCT filename) FROM {schema_table};'
    )[0][0]
    n_schema_rows = execute_query(con, f'SELECT COUNT(*) FROM {schema_table};')[
        0
    ][0]

    n_instance_files = execute_query(
        con, f'SELECT COUNT(DISTINCT filename) FROM {instance_table};'
    )[0][0]
    n_instance_rows = execute_query(
        con, f'SELECT COUNT(*) FROM {instance_table};'
    )[0][0]

    section('overview')
    print(f'  {"total files":<32} {n_schema_files + n_instance_files:>10,}')
    print(f'  {"total rows":<32} {n_schema_rows + n_instance_rows:>10,}')
    print(f'  {"schema files":<32} {n_schema_files:>10,}')
    print(f'  {"instance files":<32} {n_instance_files:>10,}')
    print(f'  {"schema rows":<32} {n_schema_rows:>10,}')
    print(f'  {"instance rows":<32} {n_instance_rows:>10,}')

    # scehma vs instance columns
    cols = execute_query(
        con,
        f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = '{schema_table}';
    """,
    )
    section(f'columns  ({len(cols)} total)')
    for (col,) in cols:
        print(f'  - {col}')

    cols = execute_query(
        con,
        f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = '{instance_table}';
    """,
    )
    section(f'columns  ({len(cols)} total)')
    for (col,) in cols:
        print(f'  - {col}')

    # schema eval library names
    lib_count = execute_query(
        con,
        f"""
        SELECT eval_library.name AS lib, COUNT(*) AS n
        FROM {schema_table}
        WHERE eval_library IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC;
    """,
    )
    section('eval library usage  (schema level)')
    for lib, n in lib_count:
        print(f'  {str(lib):<42} {n:>8,}')

    # scheme source metadata
    src_counts = execute_query(
        con,
        f"""
        SELECT source_metadata.source_type AS src, COUNT(*) AS n
        FROM {schema_table}
        WHERE source_metadata.source_type IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC;
    """,
    )
    section('source type breakdown  (schema level)')
    for src, n in src_counts:
        print(f'  {str(src):<42} {n:>8,}')

    # parameter range
    param_range = execute_query(
        con,
        f"""
        SELECT
            MIN(CAST(model_info.additional_details.params_billions AS FLOAT)),
            MAX(CAST(model_info.additional_details.params_billions AS FLOAT))
        FROM {schema_table}
        WHERE model_info.additional_details.params_billions IS NOT NULL;
    """,
    )
    section('model parameter range  (billions)')
    if param_range and param_range[0][0] is not None:
        min_p, max_p = param_range[0]
        print(f'  {"min":<32} {min_p:>10.2f}B')
        print(f'  {"max":<32} {max_p:>10.2f}B')
    else:
        print('  no params_billions data found')

    print(f'\n{SEP}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--table', default='eee', help='Table name for database'
    )

    args = parser.parse_args()
    table_name = args.table

    datastore = HUGGING_FACE_DATASTORE

    with duckdb.connect(':memory:') as con:
        con.execute('INSTALL httpfs; LOAD httpfs;')
        data_urls = read_data(datastore)

        if not data_urls:
            sys.exit(1)

        schema_urls, instance_urls = data_urls
        print_startup_summary(
            datastore=datastore,
            table_name=table_name,
            schema_urls=schema_urls,
            instance_urls=instance_urls,
        )

        if not schema_urls and not instance_urls:
            sys.exit(1)

        if schema_urls:
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {table_name}_schema AS
                SELECT * FROM read_parquet(?, union_by_name=true, filename=true)
            """,
                [schema_urls],
            )
            schema_row_count = execute_query(
                con, f'SELECT COUNT(*) FROM {table_name}_schema;'
            )[0][0]
            print_table_ready(
                name=f'{table_name}_schema',
                row_count=schema_row_count,
                file_count=len(schema_urls),
            )

        if instance_urls:
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {table_name}_instances (
                    schema_version VARCHAR NOT NULL,
                    evaluation_id VARCHAR NOT NULL,
                    model_id VARCHAR NOT NULL,
                    evaluation_name VARCHAR NOT NULL,
                    evaluation_result_id VARCHAR,
                    sample_id VARCHAR NOT NULL,
                    sample_hash VARCHAR,
                    interaction_type VARCHAR NOT NULL,
                    input JSON NOT NULL,
                    output JSON,
                    messages JSON,
                    answer_attribution JSON NOT NULL,
                    evaluation JSON NOT NULL,
                    token_usage JSON,
                    performance JSON,
                    error VARCHAR,
                    metadata JSON,
                    filename VARCHAR NOT NULL
                )
                """
            )
            for url in tqdm(
                instance_urls,
                desc='Loading instance parquet files',
                unit='file',
            ):
                available_columns = get_parquet_columns(con, url)
                instance_select_sql = build_instance_select_sql(
                    available_columns
                )
                con.execute(
                    f"""
                    INSERT INTO {table_name}_instances
                    SELECT
                        {instance_select_sql}
                    FROM read_parquet(?, filename=true)
                    """,
                    [url],
                )

            instance_row_count = execute_query(
                con, f'SELECT COUNT(*) FROM {table_name}_instances;'
            )[0][0]
            print_table_ready(
                name=f'{table_name}_instances',
                row_count=instance_row_count,
                file_count=len(instance_urls),
            )

        analyze_data(con, f'{table_name}_schema', f'{table_name}_instances')


if __name__ == '__main__':
    main()
