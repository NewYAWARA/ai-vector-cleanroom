# Changelog

此檔記錄適合公開的產品變更；私有正式素材、客戶／品牌名稱、本機路徑與內部發行證據不列入公開紀錄。

## GitHub 公開更新 — 2026-10-05

這次 GitHub 發布從 `v0.5.0-alpha` 更新到 `v3-designer-preview.4`。中間的 Beta 與 Designer Preview 編號記錄本機開發迭代，並不表示每一版都曾在 GitHub 發布。仍屬需要人工檢查的預覽版。

相較上一個公開版，主要新增完整 Illustrator 接手流程、整張／局部整理、真正原圖保存、原圖邊緣與孔洞檢查，以及逐物件可見差異提示。原生筆畫、幾何、漸層、分組及換色是承接既有能力，不是這次才首次提供。

新版以 Windows x64 / CPython 3.12 與外部專用環境為驗證目標；使用 `setup_windows.bat` 安裝、`工作台.bat` 啟動。舊入口 `install_deps.bat` 和 `workbench.bat` 保留為相同行為的入口。新版資料使用獨立的 `designer4` 目錄，不自動搬移舊資料。

各項改進和未解決問題見 [本次發布說明](release/RELEASE_NOTES.md)。沒有 Illustrator 實機完稿或設計師工時證據，不能將節點減少、測試通過或接手提示視為免修率。

## v3-designer-preview.4 — 接手診斷預覽版，待設計師驗收

- 接手時以真正原圖和最終 SVG 的可見像素比對每個交接單位，指出顏色／粗細／透明度差異、明暗變化減少與近白／透明區域上色。保留遮擋、群組、裁切、半透明合成和非方形原生畫布映射，不以固定內縮漏掉薄線。
- 移除低資訊量的通用理由；不再只因外框相交就把全圖孔洞／連通問題歸給每個碰到的物件。無法逐物件歸屬的來源問題仍保留為全圖待查。
- 診斷按原圖與 SVG 內容識別重用，記憶體、影像大小、物件數、渲染量及時間均有限制。缺少真正原圖、未完成或超出上限不當作沒有缺陷。
- 診斷不改動候選向量、不自動採用物件，也不宣稱找出所有問題。保存、匯出沿用相同的逐物件證據。
- 保留原有轉檔和驗收門檻；細長物件的新重建與覆蓋率規則仍為獨立實驗，未因局部平均誤差改善而納入正式管線。
- 預設資料與環境使用新的版本目錄，舊版保留。仍無 Illustrator 實機或設計師工時驗收。

## v3-designer-preview.3 — 預覽版，待設計師驗收

- 初次轉檔增加依原生原圖重建的獨立階段。可靠白底漸層輪廓先在原生像素取樣，再等比映射回 SVG；不把描圖器的像素階梯固定成標準答案。
- 重建需逐物件、完整場景和累積結果都通過來源檢查。原圖支持的假孔修復與真白縫、白色物件、接觸邊界分開驗證；驗證不足時保留原候選。
- 原圖的穩定實色可反駁縮圖造成的假漸層；單一量化色盤也可提出真正淡漸層候選，仍需通過既有漸層模型與場景檢查。
- 漸層搜尋另試獨立估色候選，不再完全受筆畫採用與否影響；估色取樣不刪除任何實際填色或細線，原有候選及總搜尋上限保留。
- 輪廓減點失敗時，可以保留待驗證的色彩候選；只改善既有路徑部分填色的結果維持人工確認，不冒稱完整物件或減點成功。
- 原生影像比對支援等比縮圖後短邊的整數取整，使用明確的原生畫布與等比映射；不以拉伸原圖補齊尺寸。
- 零半徑影像遮罩操作直接保留原遮罩，避免進入尺寸為一的原生影像濾鏡。
- 筆畫對照原生端點、粗細、交叉和顏色；缺少來源證據的複雜線段退回填色。原圖證明被線條遮擋的同色底圖可連回單一輪廓，減少拆碎的底圖。
- 接手頁優先定位原圖白縫填塞、斷裂或多餘淺色；短曲線反覆轉彎另列提醒，即使減點距離合格也不豁免。定位資料綁定目前 SVG，舊報告不套用新幾何。
- 原圖輪廓重建可保留少數無法通過檢查的原接縫，或在細節退步處保留原曲線；新的整體候選仍需通過原本的來源、孔洞與局部誤差檢查，不放寬驗收門檻。
- 「選取目前列表」會取代先前選取；篩選後不再連帶採用藏在列表外的物件。
- 補充混合線帽、窄縫、真小孔與淡漸層反例。原生來源驗證逾時、缺少或過期時不再顯示 designer_ready。
- 尚無真人計時或 Illustrator 匯入驗收；機器檢查與錨點數不代表可免修改完稿。

## v3-designer-preview.2 — 2026-10-05

- 新增整張自動整理、逐路徑回滾、原圖局部非退步檢查與真 SVG 漸層渲染，衍生結果另存且重建換色頁。
- 整張與局部整理另檢查合成後的透明覆蓋、孔洞與連通物件對應；轉檔的曲線整理也檢查孔洞及連通對應，採用至少 1200 像素長邊的驗證。修正白色相鄰物件在白底上看似不變，簡化後卻封洞或黏合的問題；仍不宣稱連續幾何已獲數學證明。
- 單一不透明、白底單色橢圓可從原圖亞像素輪廓重建成原生幾何；不混同於保守減點，仍為待確認提案。
- 單一白底、不透明且模型可解釋原圖的漸層圓／圓角矩形，可反算邊緣覆蓋率重建為 4／8 錨點原生形狀；保留漸層 paint 與 ID，另以原圖證據驗證，不豁免原有幾何檢查。併入邊緣殘片另驗逐處原圖誤差，憑證綁定原始 RGBA 像素、畫布、可見性與最終場景。
- 新增 working.svg：完整候選直接進 Illustrator 編輯，不需先逐個採用；原圖參考預設隱藏。
- 修正 compound 主體被其中小圓孔錯誤替換、原生漸層缺穩定 ID 導致憑證失聯或轉檔崩潰。
- 僅依原始色彩與局部渲染證據補回被描圖器完全漏掉的小元件／白色孔洞，排除漸層所有權與人工 marker 的假證據。
- 封閉曲線增加跨人工接縫合併候選；誤差門檻另檢查每個 loop 的自身尺度及包覆／交叉關係。
- 受完整場景驗證的整理可分別收集合法輪廓候選，避免單一微孔的不合法 primitive 提案牽連其他合法改善。初始漸層幾何沒有這份場景證據，維持原候選集合，不自動套用此擴充。
- 將來源支持的小假孔修復與曲線整理分成兩個事務：先保持所有其他輪廓完全相同，只補原圖及處理後影像均支持的漏色孔；其後曲線減點仍需獨立幾何、整景透明與來源檢查。分開記錄補孔與曲線減點，幾何候選延後到補孔實際提交後才計算，避免無用重複擬合。
- 補孔後整條漸層輪廓簡化被拒時，另試有界的局部分段候選；每個累積子集仍對同一份原輪廓、完整場景與原物件來源區域驗證，保留未採片段及其他輪廓的幾何。結果明示為有界搜尋的可用子集，來源、填色與最終 path 綁定驗證，不冒稱全域最少節點。
- 階段渲染改為透明 RGBA，精確操作保護白色物件，近似操作檢查新裂縫及孔洞對應。數值採樣仍不是解析幾何證明。
- 前景計分對來源／輸出使用一致的 RGBA 處理，修正白色物件在來源被計入、輸出卻被忽略而錯選帶白背景候選的問題。
- 從原始像素判斷透明度來源：原生透明圖嚴格比對透明度；白底去背產生的透明邊緣依覆蓋範圍與實際顏色比對，避免將正確細筆畫錯選成色塊，仍保護白色物件與空白背景。
- 大範圍減點未過檢查時，另以有界分塊搜尋保留可通過全部原檢查的部分；不再因單一有問題路徑而放棄所有獨立改善，不聲稱已找到數學最佳解。
- 完整撤回曲線整理且可核對原路徑資料與 ID 時，改列待人工確認，不再誤報節點經濟性證明損壞；不授予減點證明或免檢資格，也不將路徑資料核對冒稱為整個場景外觀已驗證。
- 漸層替換若造成過多額外殘留碎片，依實際被消耗區域撤回負責的候選；保留其他漸層與原有描圖。
- 漸層撤回前另以完整場景確認碎片是否原本就存在；可採只換填色的替代路徑，保留少量既有 path 的幾何與順序、共用原生漸層。要求整張透明覆蓋不變、原圖逐 path 與整個漸層區域的平均／尾端誤差通過；不冒充物件合併或幾何減點。量化色帶的處理後影像僅以每帶平均及整區尾端誤差約束，明確記錄各帶尾端差異。
- 筆畫搜尋上限依解析度有界調整，保留寬度均勻性、連通結構與長寬比門檻。
- 獨立細直線以來源覆蓋率與局部真實渲染估算次像素位置、線寬與顏色，避免把抗鋸齒邊緣誤當粗淡筆畫。
- 接手選取框支援橢圓弧、描邊與巢狀變換；Windows 工作台連接埠獨占，避免不同資料目錄服務混用。
- 框選需完整包含物件，避免小物件連帶選入背景；只看已採用時清理隱藏選取，匯出下載顯示於頁面上方。
- 原生圓與圓環補上穩定物件 ID；唯一且可見的單一物件可用直接選取證據評估，不再因不需要分組而誤判難編輯，多物件規則維持不變。
- 圓角矩形依實際幾何顯示 8 個編輯錨點，一般矩形為 4 個；工作台與結構報告使用相同計法。
- Windows 整張／局部整理子程序明確使用 UTF-8，避免中文拒絕原因因系統編碼變成通訊失敗。
- 新增 14 類設計結構、兩種解析度共 28 張的可重現合成基準，另測真實茶圖。基準包含失敗與保留結果，不代表真人省工比例或 Illustrator 驗收。

## v3-designer-preview.1 — 2026-10-04

### Illustrator 接手流程

- 新增「設計師接手」頁：原圖／SVG 比對、同步縮放平移、按物件選取、依可用邊界框選、多選與批次標記「採用／待確認／交人工」、復原／重做、狀態篩選與高節點優先排序。
- 全部物件初始為待確認。建議依現有結構與節點負擔提供，未自動採用，也不宣稱還原作者語意或已校準的信心分數。
- 提供只看已採用預覽，保留剩餘待確認與交人工數量。移除遮擋物可能露出底下幾何，仍需人工檢查。
- 新轉檔另存未去背的 `source_original.png` 作接手參考；舊結果僅有 `source_reference.png` 時明示為清理後參考圖，不當成未處理原圖。

### 保存、匯出與版本保護

- 新增工作台保存判斷；頁面以 SVG 指紋、結果位置與判斷版本管理本機草稿。同版本重新開啟會恢復未保存草稿；版本不同時先顯示工作台紀錄，同時保留衝突草稿供手動取回核對或下載 JSON。取回可復原，不會自動合併或寫回工作台。
- 保存、匯出及局部簡化要求目前 revision，使用 CAS 防止過期頁面覆蓋新判斷；缺失／過期版本或原 SVG 改變會拒絕要求。失敗不清空當前頁面判斷。
- 接手包每次另建目錄和 ZIP，包含 `accepted.svg`、`draft.svg`、`handoff.json`、manifest／decisions JSON 及 `OPEN_IN_ILLUSTRATOR.txt`。
- `accepted.svg` 只保留已採用向量；`draft.svg` 加入點陣參考與待處理提示，未採用候選隱藏保留於原結構。兩者均不被標為已完成設計驗收。
- 已保存的採用物件阻擋原地重跑；工作台改為「另開版本重跑」，保留既有結果、判斷與接手包。

### 局部輪廓減點

- 提供 0.1%、0.25%、0.5% 幾何誤差預算，只處理所選且未採用的受支援封閉單色 path；不是重新描圖或生成缺失內容。
- 單次限制 1–4 物件、最多 4 路徑，每路徑 5–512 節點、總計最多 1536；不支援筆畫、漸層／資源依賴、transform、mask、clip、filter 等上下文。
- 子程序 60 秒上限；無實際減點、超時、渲染／局部孔洞與連通結構檢查未過，均不發布新版本。不支援的選取直接拒絕，不強制改造。
- 驗證未選取幾何、樣式與堆疊保持不變；成功另建衍生結果，變更物件回到待確認，整張舊驗收分數不沿用。同物件不允許累積重複精修，需回原版比較其他誤差。

### 發行與驗證邊界

- 預覽版版本號為 `v3-designer-preview.1`；使用同名專用 venv，工作資料改在 `%LOCALAPPDATA%\AIVC\designer1`，不自動搬移 Beta.6 資料。
- 本機程式測試、瀏覽器預覽及幾何／渲染檢查不等於 Illustrator 匯入驗收、設計師完稿驗收或省時實測。尚未取得真人對照計時，不承諾省工比例或一鍵免修改率。

## Beta.6 後續變更（本預覽版承接）

### Gap-separated missing-component recovery

- 缺失小元件修復改以來源拓撲間隙判定：只處理與其他來源前景至少相隔 1 像素的單色、不透明元件，預設整批上限為 8；超出上限即整批不提案。
- render moat 內的近鄰只有在能由其他非雜訊來源 topology 元件於 1 px 對齊容差內解釋時才可接受；另以 bounded translation scan 拒絕無來源歸屬的 target-shaped shifted duplicate。若合法近鄰已侵入內圈，只能以最多 5%／32 像素且不切斷元件的局部 carve 保留分離。
- 修復維持 append-only 與原子 transaction；只有 bbox 外像素零變更、未新增內圈 strong ink、修補不接上既有或其他修補元件、目標缺失消除、未新增 topology 失敗且所有適用品質 gate 均不退步時才提交，否則完整回滾。

### Exact GPU acceleration and candidate reuse

- 新增已鎖定的 WebGPU compute backend；預設 auto 優先離散硬體，只有 exact-compatible RGB palette labeling 且 synthetic parity 通過才使用 GPU，所有 probe／OOM／device／result 異常均回原 NumPy CPU 路徑。
- GPU audit 記錄實際 adapter、backend、parity、呼叫、成功與 fallback；不以單一 kernel 倍數宣稱整體轉檔加速。VTracer、SVG 組裝、遞迴曲線與 float64 幾何 gate 仍主要使用 CPU。
- 新增 process-one-scoped exact pre-gradient cache；相同 source／background／colors／threshold／size／strokes 的候選共用 palette、stroke 與 flat state，命中深拷貝、失敗不存，候選矩陣與品質選擇規則不變。

### Persistent failure evidence

- `timed_out`、`budget_exhausted` 與 `failed` staging 改以同磁碟 rename 保存到資料根目錄 `.failed_jobs`；runtime fingerprint、GPU audit、NDJSON progress trace、最後 snapshot、logs 與 partial output 不再於失敗後消失。
- `failure_summary.json` 以事件欄位彙整候選／stage timing，未完成區段明示 lower bound；若保存本身失敗則原 staging fail closed 留在原地。成功、review、rejected 與取消 cleanup 行為不變。

### Bounded conversion and performance safeguards

- 工作台改為每張圖一個受監督子程序；預設 600 秒硬逾時，360 秒保留為內部效能優化參考，並提供真取消、階段／耗時／候選進度、私有 staging、完成收據與 SHA-256 驗證。
- 重跑期間不再預先移動正式結果；只有新結果完整通過收據與檔案驗證後才交易式提交並封存舊版。逾時、取消、候選預算用盡或 crash 均 fail closed，不提交 partial output。
- 將 marching-squares、連通元件掃描及漸層 stop-family 分組改為與舊輸出等價的有界／向量化實作；候選數、品質門檻、幾何排名與 pixel／colour 隔離規則不變。

### Windows source launcher hotfix

- 修正 source ZIP 內批次檔被封裝為 LF-only，導致 Windows `cmd.exe` 將相鄰行與標籤黏合、顯示亂碼且無法啟動的問題。
- 修正 CP65001 下 UTF-8 中文批次檔經 `goto` 後可能從多位元組字元中間恢復、產生 `� is not recognized` 的問題；使用者 launcher 改為 ASCII-only、無 `goto` 的 trampoline，繁中安裝訊息與復原交易移至 stdlib Python helper。
- GPU 鎖定依賴使用新的 `v3-codex-beta.6-gpu1` runtime key，不再原地升級舊 Beta.6 venv；setup 以工具專屬 marker／lock SHA 辨識 ownership，逐檔 probe 全部 pip metadata，損壞時以保留舊環境、失敗即 rollback 的方式重建；無 marker 的其他 venv fail closed，non-blocking setup lock 拒絕並行安裝。
- 所有公開 `.bat` 固定為 UTF-8 無 BOM、CRLF-only，且 packager 禁止非 ASCII 內容搭配 `goto`；`.gitattributes` 亦固定 checkout 行尾，避免重新發布時復發。

### Windows data-path hotfix

- 將程式碼與執行資料分離；預設 input、output 與 `output\_history` 改用較短的 `%LOCALAPPDATA%\AIVC\b6`，用來降低深層 source ZIP 路徑觸發 `WinError 206`／`Errno 2` 的風險。
- 新增 `AVC_DATA_DIR` 絕對本機路徑覆蓋；CLI 明確的 `--input`／`--output` 依舊優先，不自動搬移舊 source `input`／`output`。
- 過長上傳檔名與輸出基名改以可重現 SHA-256 尾碼縮短；同一資料根目錄加入單一 writer 保護，避免多個工作台／CLI 同時修改 input、output 或 history。
- 資料根目錄會持久保留上傳、輸出與每張圖最多 8 版 history；工作台仍只監聽 `127.0.0.1`，不上傳轉檔資料。
- 不建議將 `AVC_DATA_DIR` 設在 OneDrive、UNC／網路磁碟或多人共用資料夾。關閉所有 writer 後，可刪除 `b6` version root 移除該版工作資料。
- 已在 182 字元的深層 source root，以短 external data root 完成一張公開合成圖的實際工作台轉檔；report、SVG、ZIP 均可讀，source tree 前後未變。此變更未改動 topology、來源輪廓、renderer topology、gradient ownership 或其他品質門檻，也不擴張為任意第三方路徑皆無限制的保證。

### Distribution

- 公開發行目標改為 source-only 原始碼版本，不再把內嵌 Python runtime 當作主要交付物。
- 新增 Windows CPython 3.12 x64 的 external-venv setup／launcher 契約，以及 core／完整已驗環境的精確版本鎖定檔。
- 公開樹排除使用者素材、正式驗收素材、內部 validation、checkpoint、快取與發行暫存。

## 3.0.0-beta.6 — 2026-07-19

### Geometry and curves

- 曲線重整改為 per-path conservative frontier：候選必須先通過 topology、來源輪廓、renderer topology 與幾何尾端預算，才比較錨點、片段與節點經濟性。
- 候選、proposal 與 committed path 加入 digest／provenance guard；證據缺漏或套用前後不一致時 fail closed。
- pixel／colour similarity 不參與幾何候選排名，也不提供 identity-path 豁免。

### Gradients and editability

- 連續色場以來源空間 ownership、held-out pixels、hard-edge 與幾何預算驗證後，重建為較少且可編輯的 SVG 漸層。
- designer-quality 分別驗證 raster visual、gradient object 與 curve economy；authoritative evidence 不完整或遭變更時不標示 designer-ready。

### Safety and evidence

- 固定路徑 diagnostic runner、原子證據、lock 與 transaction guard 可避免不完整結果被提升為正式產物。
- topology、來源、renderer topology 與 gradient ownership 門檻維持保守，不以個案放寬。

### Known limits

- 文字仍輸出為輪廓 path；複雜陰影、紋理與交疊效果可能需要人工重整。
- 「80% 省工」尚未經設計師實際對照計時證實；自動品質分數與單次轉檔時間不得替代真人計時。


## Previous public release history


## v0.5.0-alpha - 2026-07-15

Isolated-component repair, light-color fidelity, and negative-space
guardrails. (Internal lineage: Codex Beta.5, built on the v0.3 engine; the
unreleased Beta.4 light-color and negative-space work is folded into this
release. Reviewed adversarially before release.)

Component topology and repair:

- Component topology schema v1: the detail diagnostic reports per-component
  coverage, fragment counts, and complete failed-component evidence
  (measurement size, viewBox, source-component labels).
- Conservative local re-trace of completely missing, isolated, opaque,
  single-color components: a separate append-only proposal SVG is rendered and
  must pass the visual gate, per-metric non-regression checks, and an exact
  outside-bbox render guard before an atomic commit. Anything ambiguous --
  multicolor, translucent, connected, partially present, edge-touching, or
  oversized -- is reported as skipped, and any failing check rolls back to the
  original bytes.
- `report.json` carries the full component-repair audit (schema
  `ai-vector-cleanroom.component-repair/v1`): status, proposal, transaction
  verdict, and per-component reasons.

Appearance and fidelity (folds in the unreleased Beta.4 work):

- Light solid objects are recovered after tracing with a conservative overlay
  and verified by a dedicated light-core coverage gate, so white text and
  white highlights are no longer hidden behind a high overall similarity score.
- Stricter structural-core threshold on low-noise sources: faint near-white
  modeling bands (RGB ~234-247 inside white glyphs) are no longer mistaken for
  independent dark structure and falsely rejected; those low-contrast pixels
  stay covered by the color and local-detail gates.
- Multi-metric visual gate: appearance is checked on overall foreground, color,
  local-detail P10, topology, and (when applicable) light-core coverage; any
  failing applicable metric is reported explicitly instead of being averaged
  away.

Negative space and geometry:

- Negative-space guardrails on circular and rectangular frames: if an inner
  hole is filled in after a geometry conversion, that conversion is rolled
  back. Candidate, metric, and final `source_reference` share one
  stroke-proven hole mask.
- Grouped-glyph counters (the enclosed holes in letters) are kept transparent
  with a conservative counter mask, so they are not painted in as light solids.

Candidates and reporting:

- Structural-risk results expand into multiple candidates; the selection
  policy is `visual_gate_tier_then_safe_dominance_then_preserve_features`.
- Paint-resource accounting fix: a reused gradient is no longer double-counted
  as a separate solid fill.

Tests: 222 (private real-logo fixtures excluded from the public suite).

## v0.3.0-alpha - 2026-07-14

Structured editing and global recolor. (Internal lineage: Codex Beta.3.2,
built on the v0.2 engine; reviewed adversarially before release.)

Local workbench:

- Multi-image upload queue fix: jobs now carry a stable id, a second upload
  stays visibly "queued" while the first is converting, and the browser keeps
  polling until every queued job reaches a terminal state (a later waiting
  file no longer looks like it vanished).
- Live result table: each finished image appears immediately, even while later
  files are still queued, instead of only after the whole batch completes.

Native geometry and structure:

- Conservative annulus detector: co-circular, same-color, same-width open
  stroke fragments become one native `<circle>` (with `stroke-dasharray`)
  only after a bidirectional 1 px raster gate; independent rollback per stage.
- Pixel-proven line/polyline nativeization: unreferenced, style/transform-free
  open `M/L/H/V` stroke paths become `<line>`/`<polyline>` only when an
  external renderer proves the RGBA pixels are identical; fails closed when no
  renderer is available. Runs after the scene-graph stage so inherited
  presentation styles are materialized first.
- Safe compound-path splitting: large paths split into independently
  selectable parts only when hole/island topology is provably preserved; a
  cubic is rewritten to a line only when control points are provably collinear
  and monotonic via exact rationals.
- Scene Graph post-process: cross-color parts become real `<g>` groups only
  when stack order and pixels are unchanged; unsafe candidates stay
  manifest-only. Every visible element gets a stable, unique ID.

Recolor and editability:

- Paint-role manifest + offline `色彩調整.html`: global recolor of fills,
  strokes and gradient stops, OKLCH-preserving, exporting explicit SVG colors
  (no tool-specific CSS variables); active-content / external-URL / injection
  guards.
- Editability audit split into three axes (`automation_readiness`,
  `redraw_complexity`, `workflow_friction`); a 5/5 structural handle count is
  never rewritten as a 5/5 human task result. Human-timing fields default to
  `not_performed`.
- Workbench Stage 2 editing-time page records SVG-handoff vs redraw seconds;
  estimated times never count toward a saving claim, and a single session
  never promotes itself to a product "80%" claim.

Quality and safety:

- `report.json` counts the final SVG DOM and keeps per-stage / recolor-role /
  actual-vs-manifest-group evidence; withdrawn stages report only committed
  counts.
- Key writes (SVG, paint-role manifest, recolor page) use atomic replace; a
  disk-write failure never leaves half an XML file.
- Opacity output (`fill-opacity` / `stroke-opacity`) and low-contrast line
  protection (`#dddddd` on white survives); original-resolution color sampling
  keeps thin lines from turning gray or fat.

## v0.2.0-alpha - 2026-07-13

- Monoline stroke reconstruction (center line + `stroke-width`), native
  `<circle>` and stroked rings, banded-ramp `<linearGradient>` reconstruction.
- Ink-ROI foreground score with bidirectional 1 px tolerance; candidate
  comparison with automatic fallback and a hard-fail floor.
- Review workbench (zoom / object list / hotspots) and a local drag-and-drop
  workbench server; Stage 1 blind-test page.
- Third-review P0 fixes: corner detection, pixel-center offset, multicolor
  line splitting, stroke-mask guard, palette merge threshold, honest
  degradation for semi-transparent sources.

## v0.1.0-alpha - 2026-07-03

- First public release. Batch PNG/JPG/WebP/BMP → grouped editable SVG with
  palette flattening and geometry regularization; honest self-check scores;
  synthetic test suite and CI.
