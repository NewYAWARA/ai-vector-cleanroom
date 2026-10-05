# 回歸測試

Preview 4 的 `test_source_object_audit.py` 包含薄線、真白色物件、透明疊色、
遮擋、群組、裁切、非方形原生映射、精準歸屬、來源變更快取失效與計算上限。
這些測試確認提示的量測與來源，不等於設計師認為每項提示都值得修改。

Designer Preview 4 另含整張整理、原圖橢圓重建、透明縫隙守門、漸層幾何證明、
完整候選接手匯出、版本衝突與不覆寫原稿測試。沒有私有素材的環境中，
需要該素材的測試會明確 skip；這不能當作已通過。

可另產生 14 類設計圖案、各 128／384px，共 28 張已知向量來源的圖例：

```bat
python tests\generate_designer_benchmark.py --destination D:\benchmark
python vector_cleanroom.py --input D:\benchmark\inputs --output D:\benchmark-results
python tests\generate_designer_benchmark.py --destination D:\benchmark --evaluate D:\benchmark-results
```

另有混合線帽、窄白縫、真小孔與淡漸層共 8 張反例，可用
`python tests\generate_designer_benchmark.py --destination D:\counterexamples --suite adversarial`
產生，依同一流程轉檔和評估。這組反例補充原有 28 張，不取代它們。

請在 setup 建立的專用 Python 環境中執行。圖例包含圓、橢圓、圓角、尖角、
孔洞、細線、等寬／變寬曲線、線性／放射漸層、相接顏色、遮擋與小點。
報告分開記錄白底可見孔洞、透明底孔洞、路徑／錨點／筆畫與單獨選取對應。
它們是結構與渲染檢查，不是語意還原、真人省工或 Illustrator 相容性的證明。
正式比較須保留執行前後的來源雜湊；不要把不同開發版本的輸出混成同一版。

本 source-only 專案不使用儲存庫內的 portable Python。請先在工具根目錄執行：

```bat
setup_windows.bat
```

setup、根目錄 launcher 與測試 launcher 共用同一個 external virtual
environment。若有設定 `AVC_VENV_DIR` 就使用該目錄；未設定時使用：

```text
%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v0.6.0-alpha
```

正式回歸測試執行：

```bat
tests\run_tests.bat
```

測試會自動產生固定圖例，從命令列完整跑一次轉換，再檢查 SVG 結構、
線寬／顏色、透明度、前景分數、自動候選與 `report.json` 欄位。
Beta.5 另檢查外觀／可編輯性雙閘門、色彩、局部細節 p10、拓撲、適用時的
淺色核心覆蓋率、實際上色資源、候選保留、淺色物件覆蓋與負空間護欄、
原生外環、compound path 安全拆分、Scene Graph 實體群組、父群組繼承樣式落地後的
逐像素直線原生化、全域換色、五項設計操作結構驗收及編輯功能的選擇政策；
SVG 內嵌 metadata 也必須與報告的兩道狀態一致。Beta.5 並驗證近白建模帶
不會誤成結構元件、component-topology schema 與完整 failed_examples 契約，
以及具 1 像素來源間隙之單色遺失元件的 deterministic 提案、預設整批 8 個上限、
非雜訊來源 topology 歸屬、target-shaped shifted duplicate／逃逸尾巴拒絕、5%／
32 像素單一連通 carve 上限、交易後無新增 gap／連接／合併、bbox 外零變更、
全 gate 非退步與 byte-exact rollback。

`run_tests.bat` 一律清除舊的 `VECTOR_TEST_REUSE` 設定，並重新建立 13 張
圖例及全部輸出，這才是正式發版驗收。

只驗證 3000px 圖上的 1px 細線修復時，可執行：

```bat
tests\run_highres_test.bat
```

它只轉換一張圖，仍會檢查前景分數至少 95、真 SVG stroke、黑色、
顯示線寬 1px±0.25px 與最多 3 個節點。

最後一次執行的輸出與主程式紀錄保留在 `tests\_last_run\`，方便定位失敗，
可直接刪除；下次測試會重新建立。

如果找不到 external venv 的 `Scripts\python.exe`，測試 launcher 會明確失敗
並要求先執行 `setup_windows.bat`，不會退回系統 Python 或舊 portable runtime。

開發中若要只重跑斷言，可自行設定 `VECTOR_TEST_REUSE=1` 後直接執行
Python unittest；框架只會接受帶有效 manifest、且核心／圖例生成器／測試版本
及 13 份完整輸出雜湊都相符的快取，否則會明確拒絕。快取簽章亦包含
局部診斷與可編輯性審計模組，這兩個模組改動時不會誤用舊報告。不可用重用模式作為
正式發版證明。

目前測試描述的是「可交給設計師的最低品質」：T／X／Y 必須輸出具有共用
接點的 3／4／3 條筆畫，圓環／方框必須是原生 primitive，透明度須保留數值，
任何尚未修好的功能都應顯示 FAIL，而不是把既有缺陷寫成預期成功。
