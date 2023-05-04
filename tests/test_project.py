from decolace.processing.project_managment import (
    AcquisitionAreaProcessing,
    ProcessingProject,
)


def test_new_project(tmp_path):
    project = ProcessingProject(project_name="test", project_path=tmp_path)
    project.write()
    assert (tmp_path / "test.decolace").exists()


def test_adding_area(tmp_path):
    project = ProcessingProject(project_name="test", project_path=tmp_path)
    project.acquisition_areas.append(AcquisitionAreaProcessing(area_name="test_area"))
    project.write()
    assert (tmp_path / "test.decolace").exists()

    ProcessingProject.read(tmp_path / "test.decolace")
    print(ProcessingProject)
