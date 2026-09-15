from geoagent.core.registry import Registry, tool_registry


def test_registry_decorator_registration():
    registry = Registry("test")

    @registry.register("demo")
    class Demo:
        pass

    assert registry.get("demo") is Demo
    assert registry.create("demo").__class__ is Demo


def test_dynamic_import_registers_tool():
    cls = tool_registry.register_from_class_path("ocr_test", "geoagent.tools.vision.ocr_tool.OCRTool")
    assert cls.__name__ == "OCRTool"
    assert tool_registry.get("ocr_test") is cls
