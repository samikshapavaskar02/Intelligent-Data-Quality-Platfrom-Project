from io import BytesIO, StringIO

import pandas as pd
import pytest

from modules.data_loader import load_dataset


class NamedBytesIO(BytesIO):
    def __init__(self, name, value=b""):
        super().__init__(value)
        self.name = name


class NamedStringIO(StringIO):
    def __init__(self, name, value=""):
        super().__init__(value)
        self.name = name


def test_load_csv():
    result = load_dataset(NamedStringIO("sample.csv", "name,age\nAda,36\n"))

    assert result.to_dict(orient="records") == [{"name": "Ada", "age": 36}]


def test_load_xlsx():
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["name", "age"])
    sheet.append(["Ada", 36])
    file = NamedBytesIO("sample.xlsx")
    workbook.save(file)
    file.seek(0)

    result = load_dataset(file)

    assert result.to_dict(orient="records") == [{"name": "Ada", "age": 36}]


def test_load_xls_uses_pandas_excel_reader(monkeypatch):
    expected = pd.DataFrame({"name": ["Ada"], "age": [36]})
    file = NamedBytesIO("sample.xls")
    calls = []

    def fake_read_excel(received):
        calls.append(received)
        return expected

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = load_dataset(file)

    assert result is expected
    assert calls == [file]


def test_load_json():
    result = load_dataset(NamedStringIO("sample.json", '[{"name":"Ada","age":36}]'))

    assert result.to_dict(orient="records") == [{"name": "Ada", "age": 36}]


@pytest.mark.parametrize("filename", ["sample.txt", "sample.xlsm", "sample.XLSX"])
def test_unsupported_extension_returns_none(filename):
    assert load_dataset(NamedBytesIO(filename)) is None
