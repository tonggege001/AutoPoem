import collections
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pypinyin import lazy_pinyin,Style
import random


EPOCH           = 20
BATCH_SIZE      = 64
LR              = 0.01
EMBEDDING_DIM   = 128
HIDDEN_DIM      = 256

MAX_LENGTH      = 100
MIN_LENGTH      = 10
MAX_WORDS       = 5000

FILE_NAME       = "poems_5.txt"
BEGIN_CHAR      = 'B'
END_CHAR        = 'E'
UNKNOWN_CHAR    = '*'

SAVE_PATH = 'torch_save_'

yunmu = ['ang1','ang2','ang3','ang4', 'eng1','eng2','eng3','eng4', 'ing1', 'ing2', 'ing3', 'ing4', 'ong1','ong2','ong3','ong4', 'an1','an2','an3','an4', 'en1','en2', 'en3', 'en4',  'in1','in2','in3','in4', 'un1','un2','un3','un4', 'ai1','ai2','ai3','ai4',
         'ei1','ei2','ei3','ei4', 'ao1','ao2','ao3','ao4', 'ou1', 'ou2','ou3','ou4','iu1','iu2','iu3','iu4','ue1','ue2''ue3''ue4' 'ui1','ui2','ui3','ui4', 'er1','er2','er3','er4', 'en1','en2','en3','en4', 'a1','a2','a3','a4', 'o1','o2','o3','o4', 'e1','e2','e3','e4', 'i1','i2','i3','i4', 'u1','u2','u3','u4', 'v1','v2','v3','v4']

def get_pin(x):
    return lazy_pinyin(x, style=Style.TONE3)


def get_suf(x):
    for i in yunmu:
        if i in x[0]:
            return i
    return None

def get_pingze(x):
    return lazy_pinyin(x, style=Style.TONE3)[0][-1]


# 在当前结果中选择一个符合韵母最高的result
def yun_max(yun_x, result):
    global train_data
    max_q = -1000000000
    num = MAX_WORDS

    for i,yun in train_data.id2yun.items():
        if yun==yun_x:
            if max_q < result[i]:
                max_q = result[i]
                num = i
    return num

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.num_layers = 2
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        input = torch.from_numpy(input)
        batch_size, seq_len = np.size(input, 0), np.size(input, 1)
        input = input.long()
        input = Variable(input)
        if hidden is None:
            h0 = torch.zeros((2, batch_size, self.hidden_dim))
            c0 = torch.zeros((2, batch_size, self.hidden_dim))
        else:
            h0, c0 = hidden

        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h0, c0))

        output = self.linear(output)
        return output


class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.poem_file = FILE_NAME
        self.poetrys = None
        self.poetrys_vector = None
        self.words = None
        self.word_size = None
        self.id2char_dict = None
        self.id2char = None
        self.char2id_dict = None
        self.char2id = None
        self.id2yun = None
        self.unknow_char = None

        self.load()
        self.create_batches()

    def load(self):
        def handle(line):
            line = line.replace(",", "")
            return line+END_CHAR

        f = open(self.poem_file, "r", encoding="utf-8")
        self.poetrys = [handle(line.strip('\n')) for line in f.readlines()]
        f.close()

        words = []
        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # zip(list1， list2...) : 将列表横向打包成元组
        # *list 可以理解为解压: 将元组集合纵向解压成两个列表
        words, _ = zip(*count_pairs)

        # 获得高频率的词汇
        word_size = min(MAX_WORDS, len(words))
        self.words = words[:word_size] + (UNKNOWN_CHAR,)
        self.word_size = len(self.words)

        # 建立词汇映射 map: char -> id， id -> char
        self.id2yun = {i:get_suf(get_pin(w)) for i,w in enumerate(self.words)}
        self.char2id_dict = {w: i for i,w in enumerate(self.words)}
        self.id2char_dict = {i: w for i,w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num, UNKNOWN_CHAR)
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys]


    def create_batches(self):
        # 不足batch的则重复数据使之成为batch
        while len(self.poetrys_vector) % self.batch_size != 0:
            r = random.randint(0, len(self.poetrys_vector)-1)
            self.poetrys_vector.append(self.poetrys_vector[r])
            self.poetrys.append(self.poetrys[r])

        self.n_size = len(self.poetrys_vector) // self.batch_size

        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i * self.batch_size:(i+1)*self.batch_size]
            length = max(map(len, batches))

            # 不够长的填充 *
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]):length] = [self.unknow_char]*r

            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)


train_data = Data(batch_size=BATCH_SIZE)

def train(param):
    global train_data
    save_path = "torch_save_"+param

    if not os.path.exists(save_path):
        model = Model(vocab_size=len(train_data.id2char_dict), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    else:
        model = torch.load(save_path)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()           # 自带softmax

    for epoch in range(EPOCH):
        for step in range(train_data.n_size):
            b_x = train_data.x_batches[step]
            b_y = train_data.y_batches[step]

            output = model(b_x)

            # 计算损失
            loss = loss_func(output.reshape(-1, MAX_WORDS+1), torch.from_numpy(b_y.reshape(-1)).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.save(model, save_path)
            print('Epoch:',epoch, "step:",step,"train loss:", loss.data)

def new_hidden():
    return torch.rand(2, 1, HIDDEN_DIM), torch.rand(2, 1, HIDDEN_DIM)


def predict_5(sentence_head, yun=""):
    global train_data
    model = torch.load(SAVE_PATH+"5")
    ans = sentence_head
    x = list(map(train_data.char2id, sentence_head))

    init_len = len(sentence_head)
    init_hidden = new_hidden()

    for cnt in range(20-init_len):
        inp = np.array(x).reshape((1, len(x)))
        out = model(inp, init_hidden)[0]

        # 考虑韵律
        if (cnt+init_len) == 9 or (cnt+init_len)==19:
            if yun[-1] == '3' or yun[-1] == '4':
                q = random.randint(1,2)
                if q == 1:
                    yun = yun[:-1]+'1'
                else:
                    yun = yun[:-1]+'2'

            out_id = yun_max(yun, out[-1]).real
            out_char = train_data.id2char(out_id)

            while out_char == '*' or out_char == 'E':
                init_hidden = new_hidden()
                out = model(inp, init_hidden)[0]
                out_id = yun_max(yun, out[-1]).real
                out_char = train_data.id2char(out_id)

        elif (cnt+init_len) == 14:
            out_id = torch.argmax(out[-1]).item().real
            out_char = train_data.id2char(out_id)
            pingze = get_pingze(out_char)

            while out_char == '*' or out_char == 'E'  or pingze=='1' or pingze=='2':
                init_hidden = new_hidden()
                out = model(inp,init_hidden)[0]
                out_id = torch.argmax(out[-1]).item().real
                out_char = train_data.id2char(out_id)
                pingze=get_pingze(out_char)
        else:
            out_id = torch.argmax(out[-1]).item().real
            out_char = train_data.id2char(out_id)
            while out_char == '*' or out_char == 'E':
                init_hidden = new_hidden()
                out = model(inp, init_hidden)[0]
                out_id = torch.argmax(out[-1]).item().real
                out_char = train_data.id2char(out_id)

        x.append(out_id)
        ans += out_char

        if (cnt+init_len) % 10 == 4:
            ans += '，\n'
            if yun == "" or yun is None:
                yun = get_suf(get_pin(out_char))

        if (cnt+init_len) % 10 == 9:
            ans += '。\n'
    print(ans)
    return ans

def deal():
    if sys.argv[1]=='5':
        predict_5(sys.argv[2])




if __name__=="__main__":
    # train("5")
    deal()
































