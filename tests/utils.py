import yaml


def assert_pipeline_yaml_equal(current_pipeline_yaml: str, expected_pipeline_yaml: str) -> None:
    """Assert that the current Haystack pipeline YAML is equal to the expected one."""
    current_pipeline = yaml.safe_load(current_pipeline_yaml)
    expected_pipeline = yaml.safe_load(expected_pipeline_yaml)
    current_pipeline["connections"] = sorted(current_pipeline["connections"], key=lambda x: x["receiver"])
    expected_pipeline["connections"] = sorted(expected_pipeline["connections"], key=lambda x: x["receiver"])
    assert current_pipeline == expected_pipeline
