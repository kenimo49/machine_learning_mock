# 独自のtensorflowをクラウドに上げて、それをtensorflow.jsとして使用する

## 概要
デブ判定機
身長と体重を入力して、肥満度を出力する

## 環境構築
python3.8.7
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## ローカルで動かす
* モデルの学習
```
python bmi_tf.py
```

## 参考
[独自学習したモデルをTensorFlow.jsで使う](https://www.mahirokazuko.com/entry/2018/08/31/154255)
