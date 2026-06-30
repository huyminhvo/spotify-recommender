from webapp import interface


def test_interface_exports_setting_scale_converter():
    assert interface.setting_scale_to_adjustment(1.0) == -0.5
    assert interface.setting_scale_to_adjustment(5.5) == 0.0
    assert interface.setting_scale_to_adjustment(10.0) == 0.5
