"""
身長と体重からクラス番号を予測するモデル
"""

from sklearn import preprocessing
import tensorflow as tf
import numpy as np

training_data = 'bmi_training.csv'
test_data = 'bmi_test.csv'


# dim次元のOne-hotベクトルを表すlistを返す関数
def one_hot(idx, dim):
    return [1 if i == idx else 0 for i in range(dim)]


# データを読み込む関数
def load_data(filename):
    with open(filename) as f:
        rows = [[float(elem.strip()) for elem in row.split(',')] for row in f.readlines() ]
        x = [row[0:2] for row in rows]  # データセットの中から, 身長と体重だけを取り出す。
        t = [one_hot(row[3], 6) for row in rows]  # データセットの中から、クラス番号だけを取り出す。分類先のクラス数は6個。

        x = preprocessing.normalize(np.array(x, dtype=np.float32)) # 正規化
        return x, t


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='x')  # 身長と体重の2次元ベクトルを受け取れるようにする。
t = tf.compat.v1.placeholder(tf.float32, shape=[None, 6], name='t')  # 「やせ」「ふつう」「肥満1度」「肥満2度」「肥満3度」「肥満4度」の6次元ベクトルを受け取れるようにする。
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob') # ドロップアウトのためのハイパーパラメータ

# ネットワーク定義
w1 = tf.Variable(tf.random_normal([2,30], mean=0.0, stddev=0.5))
b1 = tf.Variable(tf.random_normal([30], mean=0.0, stddev=0.5))
h1 = tf.matmul(x, w1) + b1
h1 = tf.nn.dropout(h1, keep_prob) # ドロップアウト。keep_probで指定した割合のノードが生き残り、それ以外のノードは値が0になる。
h1 = tf.nn.tanh(h1)

w2 = tf.Variable(tf.random_normal([30,30], mean=0.0, stddev=0.5))
b2 = tf.Variable(tf.random_normal([30], mean=0.0, stddev=0.5))
h2 = tf.matmul(h1, w2) + b2
h2 = tf.nn.dropout(h2, keep_prob) # ドロップアウト
h2 = tf.nn.tanh(h2)

w3 = tf.Variable(tf.random_normal([30,6], mean=0.0, stddev=0.5))
b3 = tf.Variable(tf.random_normal([6], mean=0.0, stddev=0.5))
y = tf.matmul(h2, w3) + b3

# 誤差関数の定義
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))

# 正解率の定義
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 推論結果の定義
predict = tf.argmax(y, 1, name='predict') # 'predict'という名前をつけた。

# 学習アルゴリズム定義
train = tf.train.AdamOptimizer().minimize(loss)

# 初期化
init = tf.global_variables_initializer()

# データセットのロード
train_x, train_t = load_data(training_data)
test_x, test_t = load_data(test_data)

with tf.Session() as sess:
    sess.run(init)
    print('Epoch\tTraining loss\tTest loss\tTraining acc\tTest acc')
    for epoch in range(3000):
        sess.run(train, feed_dict={
            x: train_x,
            t: train_t,
            keep_prob: 0.5
        })
        if (epoch+1) % 5 == 0:
            print('{}\t{}\t{}\t{}\t{}'.format(
                str(epoch+1),
                str(sess.run(loss, feed_dict={x:train_x, t:train_t, keep_prob:1.0})),
                str(sess.run(loss, feed_dict={x:test_x, t:test_t, keep_prob:1.0})),
                str(sess.run(acc, feed_dict={x:train_x, t:train_t, keep_prob:1.0})),
                str(sess.run(acc, feed_dict={x:test_x, t:test_t, keep_prob:1.0})),
            ))

    tf.saved_model.simple_save(sess, 'saved_model_debu', inputs={'x': x, 'keep_prob': keep_prob}, outputs={'predict': predict})
