from decolace.acquisition.acquisition_area import AcquisitionAreaSingle

def test_generate_acquisition_area(tmp_path):
    aa = AcquisitionAreaSingle("test", tmp_path / "test", 0.2, -1.0, 0.0)
    aa.write_to_disk()