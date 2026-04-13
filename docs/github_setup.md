# 共有PC（WSL）におけるGitHub複数アカウントの環境構築手順

このドキュメントは、1台のPC（WSL環境）を複数のユーザーで共有し、それぞれのディレクトリから別々のGitHubアカウントとして安全に開発を行うための手順書です。

## 💡 なぜこの設定が必要なのか？
Git（手元の変更履歴）の「名前設定」と、GitHub（外部サーバー）の「SSH通信鍵」は別の仕組みです。
全員で1つのSSH鍵を共有すると、GitHub側からは「すべて同じ人からのアクセス」と認識されてしまい、セキュリティやアクセス権の管理が破綻します。
そのため、**ユーザーごとに専用のSSH鍵（通信用の身分証）を発行し、`~/.ssh/config` を使って通信経路を切り替える**運用を行います。

---

# 鍵の作成と登録
## 自分用の鍵を作成
```bash
ssh-keygen -t ed25519 -C "name@users.noreply.github.com" -f ~/.ssh/id_ed25519_name
```

## 公開鍵の中身を表示してgithubにコピぺする
```bash
cat ~/.ssh/id_ed25519_name.pub
```

## sshの経路設定
```bash
nano ~/.ssh/config
```

```config
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_yamada

Host github-kuroki
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_kuroki

# ここに追加
Host github-name
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_name
```

## 接続テスト
```bash
ssh -T git@github-mine         # 自分のアカウント名が返るか確認
```

# リポジトリの準備とリモートURLの設定
## 新しくクローンするとき
```bash
git clone git@github-mine:オーナー名/リポジトリ名.git 作業フォルダ名
```

## すでにクローン済みの時
```bash
cd 自分の作業フォルダ
git remote set-url origin git@github-mine:オーナー名/リポジトリ名.git

# 変更されたか確認
git remote -v
```

# コミット用アカウントのローカル設定
```bash
cd 作業フォルダ

# 自分用のフォルダの場合
git config --local user.name "自分のアカウント名"
git config --local user.email "自分のメールアドレス(noreplyのやつ)"

# 設定の確認
git config --local --list
```
