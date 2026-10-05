# AI Vector Cleanroom

繁體中文 | [English](#english)

把 PNG、JPG、WebP、BMP 整理成**設計師可以接手的 SVG 底稿**。能用的部分先留下，難修的部分保留參考，讓設計師決定修改或重畫。

**目前版本：`v3-designer-preview.4`，Windows 原始碼預發布版。** 目標是減少設計師後續整理時間；目前尚無設計師對照計時，也尚未完成 Adobe Illustrator 實機匯入與完稿驗收。它不能保證一鍵完稿，沒有省工百分比承諾。

由 **張進逸（Shinichi Chang）** 開發與維護。MIT 授權。

## 第一次使用

1. 安裝 **Windows x64 的 CPython 3.12 x64**，包含 Python Launcher，確認 `py -3.12` 可執行。
2. 從 [Releases](https://github.com/NewYAWARA/ai-vector-cleanroom/releases) 下載此預發布版的完整原始碼並解壓縮。
3. 雙擊 `setup_windows.bat`，第一次安裝需連網下載鎖定版本的依賴。
4. 雙擊 **`工作台.bat`**，把圖片拖進本機瀏覽器頁面。
5. 轉檔完成後，按 **「設計師接手」→「匯出 Illustrator 接手包」**，解壓後先開 `working.svg`。

可視需要先按「自動整理整張圖」，也可以直接匯出。**不用先逐一按過所有物件的「採用」。** 複雜圖片可能需要數分鐘以上，工作台提供進度與取消；執行期間請保留啟動視窗。

工作台只監聽本機 `127.0.0.1`，通常使用 8765 埠，以啟動視窗顯示網址為準。轉檔不需要 API 金鑰，也不會把圖片上傳至外部服務。

## 從舊公開版改進了什麼

相較於 `v0.5.0-alpha`，這次更新把重點放在「轉完之後怎麼接手」。既有的筆畫、規則形狀、漸層、分組與換色功能持續保留；它們並非本次才新增，也不保證每張圖都能恢復成這些結構。

| 改進 | 對接手工作的用途 |
|---|---|
| 可用原圖證據重建部分輪廓、檢查假孔與白縫 | 減少把像素階梯和中間描圖錯誤當作原設計保留下來的情況 |
| 整張整理與受限的局部減點 | 能通過檢查的部分先整理；保留未能改善的部分，另存版本供比較 |
| 原圖與向量左右比對、物件選取與放大 | 直接找到需要處理的位置，不只看整張圖的平均分數 |
| Preview 4 的逐物件差異提示 | 根據真正可見的部分提醒色彩、覆蓋範圍與明暗差異，移除重複通用提醒 |
| 採用／待確認／交人工與接手包 | 可以帶著完整候選去編輯，也可以只留下已採用部分再補畫 |
| 保存、重跑與版本衝突保護 | 減少覆寫已確認成果或把過期判斷套到新圖的風險 |

提示只是待查線索，**不是必修清單**；沒有提示也不代表通過人工驗收。比對缺少原圖、超出計算上限或失敗時會明示。這次更新不宣稱所有圖片、所有區域都比舊版更好。

詳細改版說明見 [本次發布說明](release/RELEASE_NOTES.md)，歷史紀錄見 [CHANGELOG.md](CHANGELOG.md)。

## 接手包應該開哪個檔案

| 檔案 | 用途 |
|---|---|
| **`working.svg`** | 先開這個。保留完整候選向量，另含預設隱藏的嵌入點陣參考。**它不是純向量完稿。** |
| `accepted.svg` | 只保留你手動標記採用的物件。全部物件初始都是待確認，所以第一次未作判斷時，這個檔案會是空白。 |
| `draft.svg` | 已採用部分加描圖參考、隱藏候選與提示框，適合補畫；同樣不是純向量完稿。 |
| `handoff.json` 等 JSON | 待處理清單、物件資料與這次的判斷紀錄。 |
| `OPEN_IN_ILLUSTRATOR.txt` | 接手與檢查步驟。 |

在 Illustrator 需要描圖時，再顯示並鎖定參考圖。完成後移除參考圖、提示框與不需要的隱藏候選，檢查孔洞、透明度、漸層和遮擋，再另存 `.ai`。本工具不直接產生 `.ai`，也不會自動恢復原字型或可編輯文字。

## 適用範圍與目前限制

**較適合試用：** 少色、平面、邊界清楚的 icon、標籤、徽章與圖形；尤其是能接受局部人工接手的工作。

**需要更多人工處理：** 低解析度文字、細光芒、漸淡尖端、複雜漸層、互相貼合或遮擋的多色圖形。仍可能出現多餘色塊、色帶、細縫、錯色或不理想的節點。原生筆畫重建不一定成功，失敗時可能保留填色輪廓。

照片、寫實插畫、毛髮、紋理、複雜陰影與模糊效果，不是這版主要目標。圖形分組也不等於恢復原作者的圖層或設計意圖。

更少節點、更接近原圖，以及程式檢查通過，都不等於比較好改。**是否真的省工，要看設計師完成相同任務所花的時間。**

## 舊版使用者更新

請把新版解壓到新目錄，執行新版的 setup，不要覆蓋還在使用的舊版環境。這版採獨立環境與資料目錄，不自動搬移舊版資料。

| 項目 | 預設位置 |
|---|---|
| 專用 Python 環境 | `%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v3-designer-preview.4` |
| 圖片與轉檔結果 | `%LOCALAPPDATA%\AIVC\designer4` 下的 `input`、`output` |

需要重新處理時，可將原始圖片拖進新版；舊結果與已匯出的接手包會留在原處。此版本是 source-only 發行，不內嵌 Python，也不是可直接 `pip install ai-vector-cleanroom` 的套件。其他作業系統未列入此預覽版的使用驗證範圍。

自訂資料位置、批次轉檔、鍵盤操作及整理限制，見 [完整使用指南](docs/USER_GUIDE.md)。

## 歡迎提供真實工作反饋

請到 [Issues 回報](https://github.com/NewYAWARA/ai-vector-cleanroom/issues/new/choose)。比起只給整體分數，以下資訊更能決定下一步要改什麼：

- 哪些部分直接留下、修改後留下、最後仍然重畫？
- 最花時間的是找物件、修形、換色，還是整理碎片？
- 若有比較，完成相同品質的任務，工具接手與原本方法各花多久？
- 使用的版本、Windows 與 Illustrator 版本、重現步驟，以及可分享的截圖或小型合成反例。

請勿上傳客戶圖、私人圖片或未取得分享授權的作品；可以改用自己製作的最小反例。

## 開發、授權與驗證

先完成 setup，再執行 `tests\run_tests.bat`。測試說明與可自行產生的合成基準見 [tests/README.md](tests/README.md)，貢獻方式見 [CONTRIBUTING.md](CONTRIBUTING.md)。程式與渲染測試不能替代 Illustrator 實機檢查或設計師計時。

MIT 授權見 [LICENSE](LICENSE)，依賴聲明見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，作者與引用見 [AUTHORS.md](AUTHORS.md) 與 [CITATION.cff](CITATION.cff)。

## English

AI Vector Cleanroom turns flat bitmap graphics into editable SVG drafts for designer handoff. Created and maintained by **Shinichi Chang (張進逸)**. MIT licensed.

`v3-designer-preview.4` is a **Windows source-only pre-release**, targeting CPython 3.12 x64. Run `setup_windows.bat`, then `工作台.bat`. Conversion runs locally; dependency installation requires internet access.

The update adds source-aware cleanup, versioned refinement, object-level comparison and a handoff package. Open **`working.svg`** first: it includes the complete vector candidate and a hidden raster reference, so it is not vector-only final artwork. `accepted.svg` is initially empty until you explicitly accept objects.

Output still needs human review. Font recovery, complex soft effects and reliable reconstruction of every thin detail are unsupported. No designer time-saving percentage or Illustrator import/finishing validation has been established. Please [report what you kept, edited or redrew](https://github.com/NewYAWARA/ai-vector-cleanroom/issues/new/choose), using assets you are allowed to share.
