from decolace.processing.project_managment import new_project


def test_new_project(tmp_path):
    new_project("test", tmp_path)
    assert (tmp_path / "test.decolace").exists()
