# SimpleVLA
## 導入
VSCodeのマーケットプレースで`Google Colab`と検索し，Googleが出している`Colab`という名前の拡張機能を導入してください．
任意の`.ipynb`を開いて，`カーネルを選択`からColabを選べばOK．


## 実装計画
各部の最小アーキテクチャは，教育用に 1 部 = 1 ファイルへ分けています。
現状は PLAN.md の標準経路を `Genesis collect -> LeRobot load -> BC/Transformer/Flow train -> Genesis rollout eval -> Qwen connector` に寄せています。軽量な 2D/square 経路は，Genesis や Qwen が使えない環境での fallback です。

- `part1_vision.py`: Genesis render dataset，小型 CNN，分類 + 座標回帰，パッチトークン抽出，square fallback
- `part2_simulator.py`: Genesis Franka 環境，IK collector，LeRobotDataset writer/adapter，抽象 rollout 成功率評価，tiny fallback
- `part3_bc.py`: 低次元状態 MLP と画像 CNN+MLP の 1-step BC，MSE 評価，閉ループ成功率評価
- `part4_transformer_dit.py`: 条件トークン + アクショントークンを連結する単一ストリーム Transformer，チャンク実行評価
- `part5_flow_dit.py`: adaLN 付き Flow Matching DiT と Euler サンプリング，サンプル軌道の閉ループ評価
- `part6_vla_connector.py`: 凍結 Qwen3.5-0.8B の hidden state を線形 connector で DiT に接続する最小 VLA，画像+言語の閉ループ評価

`main.ipynb` は上記ファイルを順に import し，標準では Genesis/LeRobot/Qwen の接続面を使います。ローカル依存が足りない場合だけ toy fallback で同じ train/eval 関数を動かします。

### 現在の実装でできること
- `LeRobotPickPlaceDataset` は実 LeRobotDataset を読み，3〜6章の標準キー `image`, `state`, `action`, `action_chunk`, `task`, `instruction` に変換します。
- `TinyPickPlaceDataset` は同じキーを持つ fallback です。2D delta action は toy 専用で，教材標準のモデル default は Franka qpos target (`action_dim=9`) です。
- `rollout_policy(policy, env, initial_states)` により，BC / Transformer / Flow / VLA connector を Genesis 環境上の `success_rate`, `mean_final_distance` で比較できます。dataset を渡す古い呼び方は tiny fallback として残しています。
- `save_lerobot_dataset` / `save_lerobot_style_dataset` は Hugging Face LeRobot の `LeRobotDataset.create/add_frame/save_episode` を使って parquet + metadata 形式で保存します。
- `collect_genesis_franka_lerobot_dataset` は Genesis の Franka, camera render, IK waypoint, qpos 補間を使って LeRobot Dataset を収集します。`scripted_grasp=False` が標準で，cube 追従は `scripted_grasp=True` の明示 fallback に隔離しています。
- `QwenVLMBackbone` は `Qwen/Qwen3.5-0.8B` を `AutoProcessor` + `AutoModelForImageTextToText` で読み，画像+言語入力から hidden state 列を取り出します。VLA 側ではこの hidden を線形 connector で DiT チャネルへ射影します。
- `train_vla_connector_epoch` / `connector_optimizer` は VLM と DiT 本体を freeze し，connector だけを更新します。full fine-tune は `train_vla_full_epoch` の ablation に分けています。

### Genesis / LeRobot データ収集
```python
from part2_simulator import TinyPickPlaceConfig, collect_genesis_franka_lerobot_dataset

collect_genesis_franka_lerobot_dataset(
    root="data/genesis_franka_pick_place",
    repo_id="local/simple-vla-genesis-franka",
    config=TinyPickPlaceConfig(n_episodes=8, horizon=24, image_size=96),
    n_episodes=8,
    steps_per_segment=4,
    backend="cpu",  # Colab T4 なら "gpu"
    image_size=96,
    scripted_grasp=False,
)
```

### LeRobot 読み込みと Genesis 評価
```python
from part2_simulator import LeRobotPickPlaceDataset, GenesisFrankaPickPlaceEnv, rollout_policy

dataset = LeRobotPickPlaceDataset("data/genesis_franka_pick_place", repo_id="local/simple-vla-genesis-franka", chunk_size=4)
env = GenesisFrankaPickPlaceEnv(image_size=96, backend="cpu")
initial_states = [dataset.episode_initial_state(i) for i in range(4)]
metrics = rollout_policy(policy, env, initial_states)
```

### Qwen VLM 接続
```python
from part6_vla_connector import QwenVLMConfig, VLAConnectorPolicy, connector_optimizer

policy = VLAConnectorPolicy(
    qwen_config=QwenVLMConfig(
        model_id="Qwen/Qwen3.5-0.8B",
        hidden_layer=-1,
        max_condition_tokens=128,
    ),
    action_dim=9,
    chunk_size=4,
)
opt = connector_optimizer(policy, lr=1e-3)
```

### 1. 視覚観測と表現学習
物体識別や簡単な位置推定のタスクをCNNで解く．
その時獲得された潜在表現をt-SNEなどで可視化することで，表現学習について体感する．
背景や照明変化による推定精度の低下を体感する．
データ拡張による精度の向上を体感する．

### 2. 運動学とシミュレータ
Genesisを使って逆運動学ソルバーでデータ収集してみる．

### 3. 模倣学習と行動生成
前回収集したデータ収集プログラムを使ってデータを集める．
これまでに学んできたCNNと，MLPを使った1 Step生成の基本的なモデルで，うまくいくかを確かめる．
例えば低次元なPushTなら上手くいくが，ロボットマニピュレーションになると難しい，など．
観測にノイズを入れるとどうなるか検証する．

### 4. Transformer
前回作成したシンプルなMLP+CNNのPolicyにTransformerを導入し，重要な情報を取捨選択できるようにする．
それによってどのように性能や特性が変化するのかを検証する．

### 5. 生成モデル
CNN+MLP+TransformerのモデルにFlow mathingを導入する．
それによってどのように性能や特性が変化するかを分析する．

### 6. 小規模VLA構築と失敗分析
これまで開発してきた模倣学習モデルに小規模なVLMを導入し，VLA化する．
VLMはとりあえずQwen3.5-0.8Bとする．
実際に動かしてみるとうまくいかないので，その上手くいかない原因を分析する．次回ハンズオンまでの課題として，改善案を考えてきてもらう．



## 参考文献
https://techblog.exawizards.com/entry/2023/05/10/055218

https://youtu.be/wjZofJX0v4M?si=dnEC8kfNbp1h5lRO

https://github.com/poloclub/transformer-explainer


JEPAの最小実装
https://x.com/keonwkim/status/2054339552198758641?s=20
