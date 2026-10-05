# 參與改善

我們的目標是減少設計師接手向量底稿的時間。更像原圖、節點更少或測試分數更高，都不能單獨證明更省工。這仍是公開測試版。

## 設計師回饋

不會寫程式也可以參與。請用「設計師試用回饋」Issue，描述你用的版本、接手軟體，以及哪個區域保留、修改或重畫。最有幫助的是說明哪一步省時、哪一步反而增加工作；時間不確定可以寫未知，不必為了回報再做兩遍。

不需要上傳客戶原稿。可用自行繪製、可公開授權的簡單替代圖重現，或只描述問題。請勿將私人、客戶所有、商標或授權不明的圖片及其轉換 SVG 加入儲存庫。

## 程式貢獻

請針對一個具體問題提出小幅修改，附上前後行為、驗證方式與已知限制。曲線、顏色、孔洞和透明邊界常互相影響；同時保留能改善的例子與必須拒絕的反例，避免只為單張圖調參。不要以放寬驗收來掩蓋新斷口、假填色或失去的細節。

目前驗證環境是 **Windows x64、CPython 3.12.x**，依 `requirements/validated-py312.lock.txt` 安裝 18 個精確版本。其他系統與 Python 版本尚未正式驗證。先執行 `setup_windows.bat`，再執行 `tests\run_tests.bat`；測試 launcher 使用專用環境，不會自動使用任意系統 Python。

在專用環境中也可直接執行：

```powershell
python -B environment_preflight.py --full --strict-versions
python -B preflight_check.py
python -B release/package_source_beta6.py audit
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

清除 `VECTOR_TEST_REUSE` 再做正式驗證。缺少私有素材的測試會明確跳過，請分別報告通過、失敗及跳過數，不把跳過當通過。公開合成素材由 `tests/generate_fixtures.py` 產生；新增公開檔案需同步更新發行 allowlist，重新產生並檢查 `SOURCE_MANIFEST.json`。依賴升級須更新鎖定檔並重新驗證，不要只改成寬鬆版本範圍。

CI 會在 Windows x64 / Python 3.12 檢查環境、公開內容、來源封裝及解壓後測試，所有產生的工作資料放在 runner 暫存目錄。CI 不包含 Illustrator 實機操作，也不能證明設計師省時。
