import torch
from Model import Model

D_in, H, D_out = 18, 10, 4
net = Model(D_in, H, D_out)
net.load_state_dict(torch.load('./trained_model.pth'))

net.eval()

core_map = {
    "C": 1,
    "Y": 2,
    "X": 3,
    "A": 4,
    "S": 5
}
# test input:
input = ['C' 'S' 'A' 'S' 'A' 'X' 'X' 'S' 'S' 'Y' 'S' 'A' 'S' 'S' 'Y' 'C' 'Y' 'Y']
input_num = []
for row in range(len(input)):
    input_num.append([])
    for column in range(len(input[row])):
        input_num[row].append("")
        input_num[row][column]=core_map[input[row][column]]

data = torch.FloatTensor(input_num)

outputs = net(data)

classes = ("C", "Y", "X", "A", "S")

list_output = outputs.tolist()

print('Predicted: ', ' '.join('%5s' % classes[int(round(list_output[0][j]))-1]
                              for j in range(4)))
