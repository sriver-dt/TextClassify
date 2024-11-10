import torch
import torch.nn as nn
from transformers import BertModel


class BertNet(nn.Module):
    def __init__(self, num_classes, bert_base_path, freeze=True):
        super(BertNet, self).__init__()
        self.bert = BertModel.from_pretrained(bert_base_path)
        if freeze:
            for i, (name, param) in enumerate(self.bert.named_parameters()):
                if i < 6:
                    param.requires_grad = False
                    print(f'冻结bert参数{name}')
        self.output = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, mask=None):
        bert_output = self.bert(
            input_ids=x,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # bert_output['last_hidden_state'] -> [N,T,768] 每个样本，每个token的类别置信度
        # bert_output['pooler_output'] -> [N,768] 每个样本的类别置信度
        # bert_output['hidden_states'] 长度13的元组，每个元组是[N,T,768]的tensor, 每个隐藏层的输出

        # x = bert_output[0][:, 0, :]
        x = bert_output[1]  # 等价于bert_output['pooler_output']
        if x is None:
            x = bert_output[0][:, 0, :]

        return self.output(x)


# if __name__ == '__main__':
#     bert_base_chinese_path = r"D:\python\nlp\huggingface\hub\bert-base-chinese"
#     bert = Net(num_classes=12, bert_base_path=bert_base_chinese_path)
#     x = torch.randint(100, size=(2, 5))
#     o = bert(x)
