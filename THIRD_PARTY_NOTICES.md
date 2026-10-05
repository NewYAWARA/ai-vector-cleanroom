# Third-Party Notices

AI Vector Cleanroom 自有原始碼採 MIT License。下列第三方 Python 套件由 `setup_windows.bat` 安裝到儲存庫外的 virtual environment，**不因本專案採 MIT 而改變其各自授權**。

本表依 Designer Preview 2 已驗 CPython 3.12 x64 環境的套件中繼資料／隨附授權整理：

| 套件 | 已驗版本 | 套件宣告的授權 |
| --- | ---: | --- |
| cffi | 2.1.1 | MIT-0 |
| charset-normalizer | 3.5.1 | MIT |
| cssselect2 | 0.10.1 | BSD |
| freetype-py | 2.5.1 | BSD |
| lxml | 6.1.1 | BSD-3-Clause；其發行內容另含第三方授權清單 |
| NumPy | 2.3.5 | BSD-3-Clause；其發行內容另含第三方授權清單 |
| Pillow | 12.3.0 | MIT-CMU |
| pycairo | 1.29.0 | LGPL-2.1-only OR MPL-1.1 |
| pycparser | 3.0 | BSD-3-Clause |
| rendercanvas | 2.7.2 | BSD-2-Clause |
| ReportLab | 4.4.9 | BSD |
| resvg-py | 0.5.0 | MIT；內含 resvg 0.48.1 及 Rust 依賴，另有各自授權 |
| rlPyCairo | 0.4.0 | BSD |
| svglib | 2.0.2 | LGPL-3.0-or-later |
| tinycss2 | 1.5.1 | BSD |
| vtracer | 0.6.15 | MIT |
| webencodings | 0.6.1 | BSD |
| wgpu | 0.31.1 | BSD-2-Clause；wheel 隨附的 wgpu-native 採 MIT OR Apache-2.0 |

完整且具法律效力的授權文字以各套件安裝內容、上游原始碼發行版及套件 metadata 為準。NumPy、lxml、wgpu 等套件可能包含多個上游元件；重新散布 binary wheel、virtual environment 或其他組合包時，散布者必須一併檢查並履行那些授權與 notice 要求。

本 source-only 儲存庫不應提交已安裝套件、wheel 或 portable Python runtime。`requirements/*.lock.txt` 只記錄已驗版本，不包含 wheel hash，也不是第三方授權文字的替代品。

渲染器來源：[resvg-py 0.5.0 授權](https://github.com/baseplate-admin/resvg-py/blob/0.5.0/LICENSE)、[resvg 上游](https://github.com/linebender/resvg)（上游宣告 MIT OR Apache-2.0；binary 與依賴以所用版本隨附條款為準）。本原始碼包不散布該 native binary。
