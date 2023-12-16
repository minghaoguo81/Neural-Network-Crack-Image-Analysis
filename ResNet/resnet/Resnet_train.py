import torch
from Resnet_datasets import vehicledata
from Resnet_module import ResNet
from torch.utils.data import DataLoader

#加载模型
model = ResNet()
print(model)

train_on_gpu = torch.cuda.is_available()
# 使用GPU
if train_on_gpu:
    model.cuda()
#加载数据集
ds = vehicledata("../concerete_crack_images/training/Positive","../concerete_crack_images/training/Negative")
num_train_samples = ds.num_of_samples()
bs = 8
dataloader = DataLoader(ds, batch_size=bs, shuffle=True)

# 训练模型的epoch数
num_epochs = 15
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model.train()

# 损失函数
cross_loss = torch.nn.CrossEntropyLoss()
index = 0
right_sum=0
all_sum=0
for epoch in range(num_epochs):
    train_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        images_batch, type_batch =sample_batched['image'], sample_batched['type']
        if train_on_gpu:
            images_batch, type_batch = images_batch.cuda(), type_batch.cuda()
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        m_type_out_ = model(images_batch)
        type_batch = type_batch.long()

        # calculate the batch loss
        loss = cross_loss(m_type_out_, type_batch)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # update training loss
        train_loss += loss.item()

        #计算准确率
        right_sum += (m_type_out_.argmax(dim=1) == type_batch).sum().item()
        all_sum  += type_batch.shape[0]
        #每100step打印
        if index % 100 == 0:
            print('Step: {} \tTraining Loss: {:.6f} \tAccuracy: {}'.format(index, loss.item(), right_sum/all_sum))
            #print('accuracy:{}'.format(right_sum/all_sum))
        index += 1

    # 计算平均损失
    train_loss = train_loss / num_train_samples

    # 显示训练集的损失函数
    print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

# save model
model.eval()
torch.save(model, 'detect_model.pt')