# 發布來源測試版

在既有 `NewYAWARA/ai-vector-cleanroom` 儲存庫更新，保留網址、提交、Issue 與星號歷史。這次版本為 `v0.6.0-alpha`，發布日期為 2026-10-06，屬於 **source-only pre-release**。公開編號接回 v0.5 系列；誤用內部編號的 `v3-designer-preview.4` Release 與 tag 已撤下，不再列為公開下載。版本編號不是品質分數。不要重新初始化儲存庫，也不要把本機工作目錄整包上傳。

本次轉檔與接手行為沿用已完成的內部開發成果，修正公開編號、發行資訊並補齊完整雙語文件。預設專用環境為 `%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v0.6.0-alpha`；資料沿用內部相容路徑 `%LOCALAPPDATA%\AIVC\designer4`。不自動搬移或重跑資料；指向其他版本環境的 `AVC_VENV_DIR` 覆蓋須清除或改成新目錄，不可手改環境版本標記。

## 檢查要發布的內容

使用 Windows x64、CPython 3.12.x 的專用環境，依 `requirements/validated-py312.lock.txt` 安裝 18 個精確版本。沒有封裝 Python、Illustrator、客戶素材、實際客戶輸出、私有驗證報告或權杖。

```powershell
$env:VECTOR_TEST_REUSE = ''
python -B environment_preflight.py --full --strict-versions
python -B preflight_check.py
python -B release/package_source_beta6.py audit
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

`preflight_check.py` 檢查 Git 已追蹤及未被忽略的未追蹤檔案，包含 Unicode 檔名與 JSON 內容。圖片只允許發行 allowlist 中、雜湊符合 manifest 的合成測試素材；不是所有 `tests/fixtures/` 都可發布。封裝器另依明確檔案清單建立來源 ZIP，不打包整個工作目錄。兩項檢查都要通過。

## 建立與驗證下載包

將產物放在儲存庫外，例如目前使用者的暫存目錄：

```powershell
$releaseDir = Join-Path $env:TEMP 'avc-v0.6.0-alpha-release'
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
$sourceZip = Join-Path $releaseDir 'AI-Vector-Cleanroom-v0.6.0-alpha.zip'
$receipt = Join-Path $releaseDir 'SOURCE_RELEASE_RECEIPT_v0.6.0-alpha.json'
python -B release/package_source_beta6.py build --zip $sourceZip --receipt $receipt
python -B release/package_source_beta6.py verify --zip $sourceZip --receipt $receipt
```

ZIP 內的 `SOURCE_MANIFEST.json` 記錄實際封裝內容。程式、文件或 allowlist 有變更時，需從新的 ZIP 取出 manifest 同步到儲存庫，再重跑 preflight、封裝與驗證，直到儲存庫 manifest 與 ZIP manifest 一致。不要手改雜湊來略過檢查。

解壓下載包到另一個目錄，依 README 執行安裝，使用專用環境執行 `python -B environment_preflight.py --full --strict-versions` 與測試；另外以 `工作台.bat` 正常開啟介面做啟動確認。CI 也會從來源 ZIP 解壓後執行公開測試，不依賴私有素材。發布紀錄分別列出實際測試結果、跳過項目及未驗證事項，不把 CI 通過寫成 Illustrator 已驗收。

## 更新 GitHub

先檢查 `git diff --stat`、完整 diff 與 `git status --short`，只提交準備好的公開來源與文件。以正常提交更新既有儲存庫，不 force-push 主分支，保留原本的正式公開版本。誤上傳的 Preview 4 Release 與 tag 已依本次更正撤下，內部開發紀錄仍保留。

本次 `v0.6.0-alpha` 先以未公開的 Release 草稿準備；草稿所用 tag、內文與附件須在發布前對齊最終驗證提交，不將中途草稿當成已發布版本。此版本一旦正式公開，再有修改時應建立下一個版本，不重新指向已發布的 tag 或替換已發布成果。

確認尚未發布的 `v0.6.0-alpha` 草稿所用 tag 指向最終測試完成的提交，推送提交與該 tag；GitHub Release 勾選 **Pre-release**，驗證後才將草稿公開。發布內文依序放入 `release/RELEASE_NOTES.md` 的完整繁體中文與 `release/RELEASE_NOTES.en.md` 的完整英文，保留互相連結；不要以短英文摘要取代完整英文說明。可先將兩份內容合併成儲存庫外的 UTF-8 文字檔，再以該檔作為 Release body。附件使用上述已驗證來源 ZIP 與收據。CI 只做檢查，不持有發布憑證、不自動上傳 Release。

發布後確認 tag 指向驗證過的提交、附件可下載、ZIP 與收據雜湊相符。README 保留目前支援環境、安裝方式與已知限制；不要加入公開試用意見徵集或回覆承諾。

## 說明的界線

可以說「協助產生容易接手的向量底稿」，並具體列出保留、修改、重畫的工作流程。不可宣稱無損還原、任意圖片一鍵完稿、已證實節省某個百分比工時、已達理論極限或無需設計師檢查。較少節點、局部誤差和機器測試都不等於真人省時證據。
