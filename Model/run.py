import sys
import pickle
import threading
import json
import time

import numpy as np
import tensorflow as tf
from flask import Flask,request

# Declaration of global variables
input_list = []
response_dict = {}
# 进程锁
lock = threading.Lock()
params = json.load(open('params.json'))


def start():

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow

    global input_list
    global response_dict
    global lock

    x_data, _ = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    config = tf.ConfigProto(
        device_count = {'CPU':1, 'GPU':0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './model/s2ss_chatbot_anti.ckpt'

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    print("线程-模型>>模型准备就绪")

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            lock.acquire()
            if input_list:
                request = input_list.pop(0)
                ip = request.get('IP')
                infos = request.get('infos')
                print("线程-模型>>成功获取来自", ip, "的请求，内容：", infos)
                x_test = [list(infos.lower())]
                bar = batch_flow([x_test], ws, 1)
                x, xl = next(bar)
                x = np.flip(x, axis=1)

                pred = model_pred.predict(
                    sess,
                    np.array(x),
                    np.array(xl)
                )

                for p in pred:
                    ans = "".join(ws.inverse_transform(p))
                    response_dict[ip] = ans
                    print("线程-模型>>完成处理来自", ip, "的请求，返回内容：", ans)
                    break
            lock.release()
            time.sleep(0.1)


app = Flask(__name__)
@app.route('/api/chatbot', methods=['get'])
def chatbot():
    global input_list
    global response_dict
    global lock

    infos = request.args['infos']
    ip = request.remote_addr

    lock.acquire()
    input_list.append({'IP': ip, 'infos': infos})
    print("线程-HTTP>>成功入队来自", ip, "的请求，内容：", infos)
    lock.release()

    while ip not in response_dict:
        time.sleep(0.1)

    lock.acquire()
    text = response_dict.pop(ip)
    print("线程-HTTP>>成功获取属于", ip, "的响应，返回内容：", text)
    lock.release()

    return text

if __name__ == '__main__':
    app.debug = False
    model_thread = threading.Thread(target=start)
    model_thread.start()
    app.run(host='0.0.0.0', port=8000)

