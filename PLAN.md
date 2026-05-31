# SimpleVLA 実装計画

この計画は [README.md](/home/haguruma/SourceCode/simple-vla/README.md) の内容を、実装フェーズごとの作業項目に分解したものです。目的は、Genesis を使ったデータ収集から、CNN、BC、Transformer、Flow Matching DiT、Qwen VLM 接続までを、教育用に読みやすい最小構成で段階的に実装することです。

## 全体方針

- 各章の中心実装は `partX_*.py` にまとめ、Notebook は実験、可視化、操作に使う。
- 実装は「シンプルで読みやすい」ことを優先し、過剰な抽象化、広すぎるフォールバック、複雑な例外処理は避ける。
- 可能な限り同一の pick-and-place タスク、同一の Dataset/DataLoader、同一の評価指標を使い、章ごとのモデル差分を比較できるようにする。
- 実装ルートは Genesis + Franka + LeRobot Dataset の1本に固定する。CPU用、toy用、mock用、scripted用などの fallback は実装しない。
- 環境は `uv` による固定を信頼し、存在しない依存関係や未対応環境をコード側で吸収しない。
- 実行環境は GPU が使える Genesis 環境を前提とする。
- 各フェーズで「動くデモ」「学習ログ」「評価指標」「可視化」を最低1つずつ残す。
- コードは短ければ短いほど良い．

## フェーズ 1: 運動学とシミュレータ

### 対象ファイル

- `part1_simulator.py`
- `part1.ipynb`
- 参考: `ref/genesis_env.py`, `ref/make_sim_dataset.py`

### 目的

Genesis 上に Franka pick-and-place 環境を作り、IK による専門家デモを収集し、LeRobot Dataset 形式で保存する。

### 実装タスク

- Genesis 環境を構築する。
  - `show_viewer=False` のヘッドレス実行に対応する。
  - `scene.add_camera(...)` と `camera.render()` で RGB フレームを取得する。
  - `gs.init(backend=gs.gpu)` を使う。
  - Franka は Genesis 標準 MJCF から読み込む。
- pick-and-place タスクを定義する。
  - 色付き立方体を配置する。
  - 目標位置をテーブル上に配置する。
  - 物体色、開始位置、目標位置を episode ごとにランダム化する。
- IK 専門家を実装する。
  - pre-grasp
  - grasp
  - lift
  - move-to-place
  - place
  - retreat
  - 各 waypoint を IK で qpos に変換し、線形補間した qpos target をアクションとして保存する。
- 観測を保存する。
  - RGB 画像
  - proprioception / robot qpos
  - object pose
  - target pose
  - gripper 開度
  - action qpos target
  - task text
- LeRobot Dataset 書き出しを実装する。
  - `LeRobotDataset.create(...)`
  - `add_frame(...)`
  - `save_episode()`
  - `finalize()`
- ロールアウト動画を作成する。
  - IK 実行時の RGB フレームを保存する。
  - Notebook 側で mp4 として表示する。

### 検証項目

- Genesis 環境が Colab で headless 実行できる。
- 1 episode の IK 軌道が最後まで生成される。
- 保存された LeRobot Dataset を再読み込みできる。
- `LeRobotPickPlaceDataset` 経由で画像、状態、アクション、アクションチャンクが取り出せる。

### 完了条件

- `part1.ipynb` で IK 専門家のロールアウト動画を表示できる。
- Genesis 由来の LeRobot Dataset を作成し、part2 以降から読み込める。

## フェーズ 2: 視覚観測と表現学習

### 対象ファイル

- `part2_vision.py`
- `part2.ipynb`

### 目的

画像から物体クラス分類と位置回帰を学習し、後続の Transformer / DiT に渡す視覚トークンを作る。

### 実装タスク

- Dataset を整備する。
  - 事前に用意した LeRobot Dataset を読み込む。
  - Genesis レンダリングで画像、物体ラベル、物体位置ラベルを生成できるようにする。
  - Dataset は Genesis/LeRobot 由来のものだけを使う。toy dataset や mock dataset は作らない。
- 小規模 CNN を実装する。
  - 4層程度の読みやすい CNN にする。
  - pooled feature を返す。
  - patch token を返す。
  - 分類 head と位置回帰 head を持つ。
- ResNet-18 比較を実装する。
  - `torchvision.models.resnet18` を利用する。
  - 事前学習済み重みを読み込めるようにする。
  - 分類 head / 回帰 head を差し替える。
  - 必要なら patch token 相当の feature map を取り出せる API を用意する。
- 学習パイプラインを実装する。
  - classification loss
  - regression loss
  - total loss
  - train / eval loop
  - accuracy / xy error の表示
- ロバスト性実験を実装する。
  - 照明変更 test
  - 背景変更 test
  - テクスチャ/ノイズ変更 test
  - train 分布と test 分布の性能差を測る。
- データ拡張を実装する。
  - color jitter
  - random crop
  - background replacement
  - 再学習して性能低下幅が縮むか確認する。
- 表現可視化を実装する。
  - pooled feature または patch feature を抽出する。
  - t-SNE / PCA で2次元に落とす。
  - class label / object position / lighting condition で色分けする。

### 検証項目

- 小規模 CNN が Genesis/LeRobot dataset で分類と回帰を学習できる。
- Genesis/LeRobot dataset で分類と回帰が学習できる。
- 背景や照明を変えた test で性能低下が観察できる。
- データ拡張で性能低下が小さくなる。
- `encoder.encode(image)` から pooled feature と patch token が得られる。

### 完了条件

- `part2.ipynb` で学習、評価、t-SNE 可視化まで一通り実行できる。
- part3 以降が利用する視覚エンコーダ API が固定される。

## フェーズ 3: 模倣学習と1ステップ行動生成

### 対象ファイル

- `part3_bc.py`
- `part3.ipynb`

### 目的

低次元状態入力の BC と画像入力の BC を比較し、1 step 回帰の限界を体験できるようにする。

### 実装タスク

- LeRobot Dataset 用 DataLoader を用意する。
  - `LeRobotPickPlaceDataset`
  - LeRobot/Genesis episode 用の collate 関数
  - `action_dim`, `state_dim` の自動取得
- 低次元状態 BC を実装する。
  - privileged state を入力する MLP policy
  - action MSE で学習
  - closed-loop rollout で評価
- 画像 BC を実装する。
  - part2 の CNN pooled feature を使う。
  - MLP head で1 step action を回帰する。
  - action MSE と success rate の両方で評価する。
- ノイズ・分布変化実験を実装する。
  - 観測画像の照明変更
  - 背景変更
  - camera noise
  - state noise
  - success rate の低下を測る。
- 可視化を実装する。
  - 低次元 BC の成功ロールアウト動画
  - 画像 BC の失敗ロールアウト動画
  - expert trajectory と rollout trajectory の比較 plot

### 検証項目

- 低次元状態入力では Genesis pick-and-place task が解ける。
- 画像入力では MSE が下がっても closed-loop で破綻する例を示せる。
- train MSE と rollout success rate が必ずしも一致しないことを確認できる。

### 完了条件

- `part3.ipynb` で state MLP と image CNN+MLP の比較表を出せる。
- 成功例と失敗例の並列動画を作れる。

## フェーズ 4: Transformer による action chunk 生成

### 対象ファイル

- `part4_transformer.py`
- `part4.ipynb`

### 目的

画像 token と action query を Transformer に入力し、action chunk を予測する。1 step MLP との差分として、注意機構と系列出力の効果を確認する。

### 実装タスク

- 実験1: CNN + shallow Transformer + MLP を Notebook に実装する。
  - CNN feature の後段に浅い Transformer を挿入する。
  - 1 step action regression と比較する。
- 実験2: ViT encoder を Notebook に実装する。
  - CNN encoder と ViT encoder を比較する。
  - ViT attention map を可視化する。
- 実験3: action chunk Transformer を `part4_transformer.py` に実装する。
  - CNN patch token を condition token とする。
  - learned action query を `chunk_size` 個用意する。
  - `[condition tokens, action tokens]` を1本の系列として Transformer に入力する。
  - action token 部分から action chunk を出力する。
  - condition token が future action token を見ない attention mask を用意する。
- 学習と評価を実装する。
  - action chunk MSE
  - first-action receding horizon rollout
  - chunk 全体を実行する rollout
  - chunk boundary jitter の測定
- attention 可視化を実装する。
  - action token から image patch への attention を取り出す。
  - 対象物や背景への注目がどう変化するかを heatmap で表示する。

### 検証項目

- `ChunkTransformerPolicy` が `image -> [chunk, action_dim]` を返す。
- action chunk MSE が学習で下がる。
- 1 step BC と比較して軌道が滑らかになるか確認できる。
- attention map を画像上に重ねて表示できる。

### 完了条件

- `part4.ipynb` で実験1, 2, 3を比較できる。
- Transformer policy の rollout 評価と attention 可視化が実行できる。

## フェーズ 5: Flow Matching DiT

### 対象ファイル

- `part5_flow.py`
- `part5.ipynb`

### 目的

Transformer action chunk model を Flow Matching DiT に拡張し、MSE 回帰では平均化して失敗するマルチモーダル行動生成を扱えるようにする。

### 実装タスク

- Flow Matching の学習目標を実装する。
  - clean action chunk
  - Gaussian noise
  - interpolation `x_t = (1 - t) * noise + t * clean`
  - target velocity `clean - noise`
  - velocity MSE
- DiT block を実装する。
  - action chunk を token に埋め込む。
  - 画像 patch token を condition token として連結する。
  - timestep embedding を作る。
  - adaLN で Transformer block を時刻条件付けする。
  - velocity head で action chunk 形状に戻す。
- 推論 sampler を実装する。
  - noise から開始する。
  - 8から10 step 程度の Euler integration を行う。
  - sampled action chunk を rollout に渡す。
- 2D 速度場の数式可視化を Notebook に実装する。
  - 2D 分布で速度場を表示する。
  - noise からデータ分布へ移動する様子を描画する。
- マルチモーダル実験を設計する。
  - 同じ目標を達成する2通りの把持アプローチを用意する。
  - MSE 回帰版は平均行動で失敗しやすいことを示す。
  - Flow 版はどちらかの mode を sample して成功することを示す。
- 可視化を実装する。
  - 複数 sample の action trajectory
  - noise から action に収束する過程
  - MSE 版と Flow 版の並列 rollout 動画

### 検証項目

- `FlowMatchingDiT.forward(image, noisy_action, t)` が velocity を返す。
- `flow_matching_loss` が学習で下がる。
- `sample(image)` が action chunk を返す。
- 同一観測から複数サンプルを生成し、多様性を数値化できる。

### 完了条件

- `part5.ipynb` で Flow Matching の2D速度場可視化と Genesis task 評価を実行できる。
- MSE Transformer と Flow DiT の比較結果を表と動画で示せる。

## フェーズ 6: 小規模 VLA 構築と失敗分析

### 対象ファイル

- `part6_vla.py`
- `part6.ipynb`
- 参考: `docs/qwen-vla.pdf`

### 目的

完成した action chunk / DiT policy の condition prefix を、CNN feature から Qwen VLM hidden state に差し替え、小規模 VLA として動かす。さらに、うまくいかない原因を分析する。

### 実装タスク

- Qwen VLM wrapper を実装する。
  - `AutoProcessor`
  - `AutoModelForImageTextToText`
  - 画像 + 言語指示を入力する。
  - hidden states を取得する。
  - 使用する hidden layer を選べるようにする。
  - condition token 数を `max_condition_tokens` で制限する。
- 線形 connector を実装する。
  - Qwen hidden dim から DiT/Transformer dim へ射影する。
  - まず学習対象は connector 1層のみに限定する。
  - VLM 本体は freeze する。
  - DiT body も原則 freeze し、必要に応じて ablation として connector + DiT を学習する。
- VLA policy を構築する。
  - Qwen hidden token を condition prefix とする。
  - action token または noisy action token を後続に連結する。
  - part4/part5 と同じ attention 系列として処理する。
- 言語条件付きデータを用意する。
  - 複数色の立方体を配置する。
  - 日本語指示を作る。
    - `赤い立方体を掴め`
    - `青い立方体を掴め`
    - `緑の立方体を目標位置へ置け`
  - 英語指示との比較も任意で行う。
- 評価を実装する。
  - seen color / seen instruction
  - seen color / paraphrased instruction
  - unseen background
  - unseen object layout
  - wrong instruction への挙動確認
- 失敗分析を実装する。
  - hidden layer ごとの性能比較
  - connector-only 学習の限界
  - VLM hidden が物体位置を十分に保持しているかの probing
  - 言語指示を変えたときの action 差分
  - 画像のみ / 言語のみ / 画像+言語の ablation

### 検証項目

- Qwen forward check で hidden shape が確認できる。
- `VLAConnectorPolicy` が `image + instruction -> action_chunk` を返す。
- connector-only 学習が実行できる。
- 言語指示を変えると出力 action が変化するか確認できる。
- OOD 条件で失敗例を収集できる。

### 完了条件

- `part6.ipynb` で VLA の forward、学習、rollout、失敗分析を実行できる。
- 次回ハンズオンに向けて、改善案を議論できるだけの失敗ログと可視化が揃う。

## Notebook 作成計画

### `part1.ipynb`

- Genesis の初期化
- Franka scene の表示/レンダリング
- IK waypoint の確認
- 専門家ロールアウト動画
- LeRobot Dataset 保存と再読み込み

### `part2.ipynb`

- dataset 読み込み
- 小規模 CNN 学習
- ResNet-18 学習
- ロバスト性評価
- データ拡張比較
- t-SNE/PCA 可視化

### `part3.ipynb`

- state MLP BC 学習
- image CNN+MLP BC 学習
- MSE と success rate の比較
- 観測ノイズ評価
- 成功/失敗 rollout 動画

### `part4.ipynb`

- CNN + shallow Transformer 実験
- ViT 実験
- `ChunkTransformerPolicy` 学習
- chunk rollout 評価
- attention heatmap 可視化

### `part5.ipynb`

- Flow Matching 2D velocity field visualization
- `FlowMatchingDiT` 学習
- sampling 可視化
- MSE Transformer との比較
- マルチモーダル行動生成デモ

### `part6.ipynb`

- Qwen hidden extraction check
- connector-only 学習
- VLA rollout
- 日本語指示への追従性確認
- OOD 評価
- 失敗分析と改善案メモ

## 優先順位

1. `part1_simulator.py` の Genesis + LeRobot Dataset 収集ルートを安定化する。
2. `part2_vision.py` に ResNet-18 と可視化用 feature extraction を追加する。
3. `part3_bc.py` と `part3.ipynb` で baseline 評価表を作る。
4. `part4_transformer.py` の action chunk Transformer を Notebook から使いやすく整える。
5. `part5_flow.py` の Flow DiT を比較実験できる形にする。
6. `part6_vla.py` の Qwen wrapper と connector-only 学習を最小実行まで通す。
7. 各 Notebook に可視化、動画、評価表を追加する。

## リスクと対策

- Genesis の grasp physics が不安定な可能性がある。
  - fallback は入れず、gripper 制御、接触条件、waypoint、成功判定を調整して本実装を安定化する。
- 実行時間が長くなる可能性がある。
  - fallback は入れず、episode 数、画像解像度、batch size、学習 epoch 数を設定値として小さくできるようにする。
- LeRobot Dataset API のバージョン差分が出る可能性がある。
  - Dataset adapter を薄く保ち、Notebook 側に LeRobot 依存を散らさない。
- Qwen3.5-0.8B のモデル ID や Transformers 対応状況が変わる可能性がある。
  - `uv.lock` と `pyproject.toml` で実行環境を固定し、コード内では別モデルへの fallback を実装しない。
  - 最初に hidden shape を確認し、期待形状と異なる場合は実装または依存関係を修正する。
- connector-only VLA は性能が出ない可能性が高い。
  - これはフェーズ6の狙いに含め、失敗分析、probe、ablation を成果物にする。

## 最終成果物

- Genesis Franka pick-and-place の IK 専門家データ収集スクリプト
- LeRobot Dataset 形式のデモデータ
- CNN / ResNet による視覚表現学習
- state BC と image BC の比較実験
- action chunk Transformer policy
- Flow Matching DiT policy
- Qwen hidden を条件 prefix に使う小規模 VLA
- 各フェーズの Notebook、評価表、可視化、rollout 動画
