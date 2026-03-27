from types import SimpleNamespace

from every_eval_ever.converters.common.utils import (
    extract_file_uuid_from_detailed_results,
)


def test_extract_file_uuid_from_detailed_results_parses_uuid_samples_jsonl():
    uuid_value = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'
    fake_log = SimpleNamespace(
        detailed_evaluation_results=SimpleNamespace(
            file_path=f'/tmp/some_dataset/some_model/{uuid_value}_samples.JSONL'
        )
    )

    assert extract_file_uuid_from_detailed_results(fake_log) == uuid_value


def test_extract_file_uuid_from_detailed_results_returns_none_without_path():
    fake_log = SimpleNamespace(
        detailed_evaluation_results=SimpleNamespace(file_path=None)
    )

    assert extract_file_uuid_from_detailed_results(fake_log) is None
