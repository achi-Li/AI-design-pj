from model import myGPT,myGPTConfig
import numpy as np
import sentencepiece as spm
import sys
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from data_set import load_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate = 1e-3
max_iters = 50000

train_data = np.memmap('train.dat', dtype=np.int32, mode='r')
target_data = np.memmap('target.dat', dtype=np.int32, mode='r')


def get_batch(split, config:myGPTConfig):
    mydata = train_data if split == 'train' else target_data
    ix = torch.randint(len(mydata) - config.time_sequence_length, (config.batch_size,))
    x = torch.stack([torch.from_numpy((mydata[i:i + config.time_sequence_length]).astype(np.int32)) for i in ix])
    y = torch.stack([torch.from_numpy((mydata[i + 1:i + 1 + config.time_sequence_length]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def train(config:myGPTConfig, model:myGPT):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    #plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    line, = ax.plot([], [], label='Training Loss')
    plt.legend()

#    with tqdm(total=max_iters, desc="Training Progress", ncols=100) as pbar:
    for iter_num in range(max_iters):
        optimizer.zero_grad()

        xb, yb = get_batch('train', config)

        logits, loss = model(xb, yb)
        losses.append(loss.item())

        if (iter_num + 1) % 1000 == 0:
            print(f"[train_info] iter:{iter_num + 1:5d}, loss:{loss.item():5.3f}")

            #line.set_data(range(len(losses)), losses)
            #ax.relim()
            #ax.autoscale_view()
            #fig.canvas.draw()
            #fig.canvas.flush_events()

        loss.backward()
        optimizer.step()
        #pbar.update(1)

    #print("train finish")
    #plt.ioff()  # 关闭实时模式
    line.set_data(range(len(losses)), losses)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    plt.savefig("./loss.jpg")
    #plt.show()  # 显示最终图形
    print(f"final loss: {loss.item()}")


def main():
    config = myGPTConfig()

    GPTmodel = myGPT(config).to(device)
    token_model_file = "mydata.model"

    flag, sp = load_tokenizer(token_model_file)
    if not flag:
        print(f"load tokenizer model from: {token_model_file} failed")
        sys.exit(1)

    # 文本生成部分的可视化
    user_inputs = ["韩立眉头皱了皱，但忽然有所感应",
                   "听了这话，韩立有点诧异的转过身",
                   "毕竟韩立第二元婴，一看就是",
                   "城门上一阵骚动，当即有一名身穿",
                   "血丝金光交织中，金色飓风",
                   "啼魂兽对着一片黑暗的森林吼叫",
                   "墨彩环在梦中见到韩立",
                   "韩立在古老的遗迹中穿梭",
                   "南宫婉在修炼中遇到瓶颈",
                   "噬金虫围绕着一块奇异的石头",
                   "啼魂兽在一片废墟中嗅着什么"]

    for user_input in user_inputs:
        context = torch.tensor([sp.encode(user_input)], dtype=torch.int32, device=device)
        print(f"输入: {user_input}")
        gpt_output = GPTmodel.generate(context, max_new_tokens=200)[0].tolist()

        for token_id in gpt_output:
            generated_text = sp.decode([token_id])
            print(generated_text, end='', flush=True)
            #time.sleep(0.1)
        print("\n" + "="*50)

    train(config, GPTmodel)

    for user_input in user_inputs:
        context = torch.tensor([sp.encode(user_input)], dtype=torch.int32, device=device)
        print(f"输入: {user_input}")
        gpt_output = GPTmodel.generate(context, max_new_tokens=200)[0].tolist()

        for token_id in gpt_output:
            generated_text = sp.decode([token_id])
            print(generated_text, end='', flush=True)
            time.sleep(0.1)
        print("\n" + "="*50)


if __name__ == '__main__':
    main()