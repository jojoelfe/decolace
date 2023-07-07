from decolace.processing.project_managment import (
    AcquisitionAreaPreProcessing,
    ProcessingProject,
    MatchTemplateRun,
)


def test_new_project(tmp_path):
    project = ProcessingProject(project_name="test", project_path=tmp_path)
    project.write()
    assert (tmp_path / "test.decolace").exists()


def test_adding_area(tmp_path):
    project = ProcessingProject(project_name="test", project_path=tmp_path)
    project.acquisition_areas.append(
        AcquisitionAreaPreProcessing(area_name="test_area")
    )
    project.write()
    assert (tmp_path / "test.decolace").exists()

    read_project = ProcessingProject.read(tmp_path / "test.decolace")
    assert read_project.acquisition_areas[0].area_name == "test_area"
    assert read_project.acquisition_areas[0].decolace_acquisition_area_info_path is None

    project = ProcessingProject(project_name="test", project_path=tmp_path)
    project.acquisition_areas.append(
        AcquisitionAreaPreProcessing(
            area_name="test_area", view_image_path="/tmp/test.mrc"
        )
    )
    project.match_template_runs.append(
        MatchTemplateRun(
            run_name="test_run",
            run_id=1,
            template_path="/tmp/test.mrc"
        )
    )
    project.write()
    read_project = ProcessingProject.read(tmp_path / "test.decolace")
    assert read_project.acquisition_areas[0].area_name == "test_area"
    assert str(read_project.acquisition_areas[0].view_image_path) == "/tmp/test.mrc"
    assert read_project.match_template_runs[0].run_name == "test_run"
    assert str(read_project.match_template_runs[0].template_path) == "/tmp/test.mrc"

    
    
            
            
