# Designer Preview 4 改版說明

版本：`v3-designer-preview.4` · 2026-10-05 · **預發布版**

這次從舊公開版 `v0.5.0-alpha` 更新，目標是讓設計師更容易接手、保留可用成果並補畫難處。它不是一鍵完稿版，也不是「省工率已驗證」的版本。版本號沿用後續開發系列，不代表成熟度已到穩定版。

## 設計師會用到的變更

- **多一個完整接手流程。** 原圖與向量左右比對，能選取、放大與批次標記「採用／待確認／交人工」。不必先替所有物件分類，就能匯出完整候選去 Illustrator 接著改。
- **整理後另留版本。** 整張自動整理與受限的局部減點，讓可處理部分先改善；未通過檢查的部分保留，已採用物件受到保護。重跑、保存與多視窗操作加入版本核對，避免過期操作覆寫較新的判斷。
- **回到原圖判斷部分輪廓與孔洞。** 對支援的情況，依原圖邊緣提出輪廓重建，並檢查假孔、白縫與新裂縫。中間描圖結果不再是唯一依據。證據不足或會破壞其他區域時保留原候選。
- **分開評估顏色與形狀。** 支援條件內，輪廓無法安全簡化時，仍可保留通過來源檢查的漸層填色改善；不把只改填色算成合併物件成功。
- **Preview 4 改善問題定位。** 依每個接手物件真正可見的部分對照原圖，提醒色彩或覆蓋範圍不同、留白處多餘上色、明暗變化遺失。移除重複通用提醒；全圖結構疑點另列，不因外框相交就怪到每個大物件上。選取框也不再把細線染色，方便比色。
- **Windows 安裝與資料分離。** 使用專用 CPython 3.12 x64 環境與鎖定依賴，工作資料放在獨立的本機目錄。轉檔有進度、取消與時間上限；失敗或超時不當成完整成果發布。

原版已有的 SVG 筆畫、規則形狀、漸層、分組、換色與校稿功能繼續保留。本次沒有把它們全算成新增，也不保證每個物件都能變成可調線寬的筆畫或單一漸層。

Preview 4 本身主要更新接手診斷，沒有把研究中的細長物件重建加入預設管線，也沒有為了讓候選通過而直接放寬既有來源檢查。

## 匯出方式變了，先開 working.svg

- **`working.svg`**：完整候選向量，加上預設隱藏的嵌入原圖參考。適合直接開始人工編輯；它不是純向量完稿。
- **`accepted.svg`**：只有手動採用的部分。初始全部物件都是待確認，因此未作判斷的第一次匯出會是空白。
- **`draft.svg`**：採用部分、點陣參考、隱藏候選和定位提示，方便補畫，同樣不是純向量完稿。
- 接手包另附清單、判斷 JSON 和 `OPEN_IN_ILLUSTRATOR.txt`。每次匯出另存一份，不覆寫前一份接手包。

完成後請移除參考圖、提示框及不需要的隱藏候選，檢查透明度、孔洞、漸層和遮擋，再另存 `.ai`。工具不直接輸出 `.ai`，文字通常仍是輪廓，無法可靠恢復原字型。

## 安裝與從舊版更新

1. 準備 Windows x64 與 CPython 3.12 x64，含 Python Launcher。
2. 下載完整原始碼到新目錄，執行 `setup_windows.bat`；第一次需連網。
3. 執行 `工作台.bat`，將原始圖片拖進新版。

預設環境為 `%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v3-designer-preview.4`，資料為 `%LOCALAPPDATA%\AIVC\designer4`。**不自動搬移、重跑或覆寫舊版資料。** 請保留舊輸出與接手包；不要把新程式直接覆蓋到仍在使用的舊環境。

本版不附 Python runtime，其他平台尚未列入使用驗證範圍。操作細節見 [README](https://github.com/NewYAWARA/ai-vector-cleanroom/blob/v3-designer-preview.4/README.md) 與 [完整使用指南](https://github.com/NewYAWARA/ai-vector-cleanroom/blob/v3-designer-preview.4/docs/USER_GUIDE.md)。

## 已知限制與驗證範圍

- 文字內部、細光芒、淡尖端、色帶、相接邊緣與局部顏色仍可能需要人工修改或重畫。有些區域可能較舊版退步，不能只看整體平均誤差。
- 提示可能漏報或過多，排序也不等於設計重要性或修改工時。沒有提示、較少節點或機器標記通過，都不代表可直接交件。
- 複雜柔邊、陰影、紋理、照片和原字型恢復不是這版主力。分組也不代表還原作者的語意或原始圖層。
- 凍結的開發版本機驗證紀錄為 **975 項 unittest，5 項跳過，0 失敗**；另重跑 36 張合成基準，33 張機器判定接受、3 張要求人工確認。36 張已知正確圖的診斷負控制未觸發新差異提示。這些範圍有限，不能推論所有真圖都沒有誤報或退步。
- **尚無 Illustrator 實機匯入／完稿驗收，也尚無設計師對照計時。** 測試數量不是品質分數或省工比例；公開提交的檢查另以該次 CI 與發行檢查為準。

## 這次最需要的反饋

請到 [Issues](https://github.com/NewYAWARA/ai-vector-cleanroom/issues/new/choose) 告訴我們，哪些部分保留、修改或重畫，以及哪一步最花時間。若有對照，請記錄完成相同要求時，工具接手與原本方法各花多久。下一步以這些實際負擔決定優先順序。

可以附可公開的最小反例；不要提交客戶、私人或授權不明的圖稿。正式素材、內部研究輸出與本機驗證資料不隨公開原始碼發布。

## English summary

This pre-release updates the previous public `v0.5.0-alpha` with source-aware cleanup, versioned refinement and a designer handoff workflow. Preview 4 adds visible-object comparison against the original image, replacing generic warnings with specific review hints.

Windows x64 and CPython 3.12 x64 are the supported setup target. Run `setup_windows.bat`, then `工作台.bat`. Existing data is not migrated automatically.

Open `working.svg` first. It contains the full candidate and a hidden raster reference; it is not vector-only final artwork. `accepted.svg` remains empty until objects are explicitly accepted. Human review is required: neither Illustrator finishing nor designer time savings has been validated. Please report what you kept, edited or redrew.
